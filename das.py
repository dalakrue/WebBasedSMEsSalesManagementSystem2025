from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import SignupForm, ProductForm, ReportForm
from .models import Profile, Product, Cart, Order, OrderItem, Report
from decimal import Decimal
from django.contrib.auth.models import User
from .forms import ProductImportForm
from django.contrib import messages
from .forms import ProductForm
from .models import Profile, Product, Cart, Order, OrderItem, Report, ProductChangeRequest, Category
from .models import Product, Cart, Order, OrderItem
import io
import csv
# User/Seller Signup & Login

def signup_view(request):
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            password = form.cleaned_data['password']
            role = form.cleaned_data['role']
            address = form.cleaned_data.get('address')
            phone_number = form.cleaned_data.get('phone_number')

            user.set_password(password)
            user.save()

            Profile.objects.create(
                user=user,
                role=role,
                address=address if role == 'seller' else '',
                phone_number=phone_number if role == 'seller' else ''
            )

            return redirect('login')
    else:
        form = SignupForm()
    return render(request, 'core/signup.html', {'form': form})

def login_view(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user:
            login(request, user)

            if user.is_superuser:
                request.session['role'] = 'admin'
                return redirect('dashboard_admin')

            try:
                role = user.profile.role
                request.session['role'] = role
                return redirect(f'dashboard_{role}')
            except Profile.DoesNotExist:
                logout(request)
                return render(request, 'core/login.html', {'error': 'Profile not found'})
        else:
            return render(request, 'core/login.html', {'error': 'Invalid credentials'})
    return render(request, 'core/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

# Admin Static Login System

def admin_login(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username == 'admin' and password == 'admin123':
            request.session['is_admin'] = True
            return redirect('dashboard_admin')
        else:
            return render(request, 'core/admin_login.html', {'error': 'Invalid admin credentials'})
    return render(request, 'core/admin_login.html')

def admin_logout(request):
    request.session.flush()
    return redirect('admin_login')

# Dashboards

# Buyer dashboard - hide out-of-stock products
@login_required
def dashboard_user(request):
    products = Product.objects.filter(approved=True, quantity__gt=0)
    return render(request, 'core/dashboard_buyer.html', {'products': products})

@login_required(login_url='login')
def dashboard_seller(request):
    products = Product.objects.filter(seller=request.user)
    return render(request, 'core/dashboard_seller.html', {'products': products})


def dashboard_admin(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    products = Product.objects.all()

    # Fetch pending edit and delete requests
    edit_requests = ProductChangeRequest.objects.filter(request_type='edit', approved=False)
    delete_requests = ProductChangeRequest.objects.filter(request_type='delete', approved=False)

    # Optional summary counts for dashboard cards
    total_products = products.count()
    pending_edit_requests = edit_requests.count()
    pending_delete_requests = delete_requests.count()

    context = {
        'products': products,
        'edit_requests': edit_requests,
        'delete_requests': delete_requests,
        'total_products': total_products,
        'pending_edit_requests': pending_edit_requests,
        'pending_delete_requests': pending_delete_requests
    }

    return render(request, 'core/dashboard_admin.html', context)

# Seller Product Features

@login_required(login_url='login')
def add_product(request):
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)  # Important: include request.FILES
        if form.is_valid():
            product = form.save(commit=False)
            product.seller = request.user  # assign the logged in user as seller
            product.save()
            messages.success(request, "Product added successfully!")
            return redirect('dashboard_seller')
        else:
            # You can print to console or log errors to debug
            print(form.errors)
    else:
        form = ProductForm()

    return render(request, 'core/add_product.html', {'form': form})
# Admin Approval

def approve_product(request, product_id):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    product = get_object_or_404(Product, id=product_id)
    product.approved = True
    product.save()
    return redirect('dashboard_admin')

# Admin Delete Product

def delete_product(request, product_id):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    if request.method == 'POST':
        product = get_object_or_404(Product, id=product_id)
        product.delete()
    return redirect('dashboard_admin')

# Buyer: Cart + Checkout

# Add product to cart
@login_required
def add_to_cart(request, product_id):
    product = get_object_or_404(Product, id=product_id, approved=True)
    if product.quantity < 1:
        messages.error(request, f"{product.name} is out of stock.")
        return redirect('dashboard_user')

    cart_item, created = Cart.objects.get_or_create(user=request.user, product=product)
    if not created:
        cart_item.quantity += 1
        cart_item.save()
    messages.success(request, f"Added {product.name} to cart.")
    return redirect('dashboard_user')

# View cart
@login_required
def cart_view(request):
    cart_items = Cart.objects.filter(user=request.user)
    for item in cart_items:
        item.subtotal = item.quantity * item.product.price
    total = sum(item.subtotal for item in cart_items)
    return render(request, 'core/cart.html', {'cart_items': cart_items, 'total': total})

# Update quantity
@login_required
def update_cart_quantity(request, item_id):
    cart_item = get_object_or_404(Cart, id=item_id, user=request.user)
    if request.method == 'POST':
        new_quantity = int(request.POST.get('quantity', 1))
        if new_quantity > 0:
            cart_item.quantity = new_quantity
            cart_item.save()
    return redirect('cart')

# Remove item from cart
@login_required
def remove_from_cart(request, item_id):
    cart_item = get_object_or_404(Cart, id=item_id, user=request.user)
    cart_item.delete()
    messages.success(request, "Item removed from cart.")
    return redirect('cart')

# Checkout
@login_required
def checkout(request):
    cart_items = Cart.objects.filter(user=request.user)
    total = sum(Decimal(str(item.product.price)) * item.quantity for item in cart_items)

    if request.method == 'POST':
        # Check stock
        for item in cart_items:
            if item.product.quantity < item.quantity:
                return render(request, 'core/checkout.html', {
                    'total': total,
                    'error': f"Not enough stock for {item.product.name}."
                })

        # Create order
        order = Order.objects.create(user=request.user, total_price=total)
        for item in cart_items:
            OrderItem.objects.create(
                order=order,
                product=item.product,
                quantity=item.quantity,
                seller=item.product.seller
            )
            # Update product quantity
            product = item.product
            product.quantity -= item.quantity
            product.save()

        cart_items.delete()
        return render(request, 'core/checkout.html', {'success': True})

    return render(request, 'core/checkout.html', {'total': total})




@login_required(login_url='login')
def buyer_orders(request):
    orders = Order.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'core/buyer_orders.html', {'orders': orders})

@login_required(login_url='login')
def seller_orders(request):
    order_items = OrderItem.objects.filter(seller=request.user).select_related('order').order_by('-order__created_at')
    return render(request, 'core/seller_orders.html', {'order_items': order_items})

def admin_orders(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    orders = Order.objects.all().order_by('-created_at')
    return render(request, 'core/admin_orders.html', {'orders': orders})

@login_required
def report_issue(request):
    user = request.user
    # Check role based on profile.role field
    role = 'buyer' if getattr(user.profile, 'role', '') == 'user' else 'seller'
    if request.method == 'POST':
        form = ReportForm(request.POST)
        if form.is_valid():
            report = form.save(commit=False)
            report.reporter = user
            report.role = role
            report.save()
            return redirect('dashboard_user' if role == 'buyer' else 'dashboard_seller')
    else:
        form = ReportForm()

    return render(request, 'core/report_form.html', {'form': form, 'role': role})

def view_reports_admin(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    reports = Report.objects.all().order_by('-submitted_at')
    return render(request, 'core/admin_reports.html', {'reports': reports})

def admin_logout(request):
    request.session.flush()  # Clears all session data
    return redirect('admin_login')

def import_products_view(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    if request.method == 'POST':
        form = ProductImportForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            if not csv_file.name.endswith('.csv'):
                messages.error(request, "Please upload a CSV file.")
                return redirect('import-products')

            decoded_file = csv_file.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)
            reader = csv.DictReader(io_string)

            created_count = 0
            skipped_count = 0

            for row in reader:
                try:
                    # Get seller
                    seller_username = row.get('seller')
                    if not seller_username:
                        skipped_count += 1
                        continue
                    seller = User.objects.get(username=seller_username)

                    # Get category by name
                    category_name = row.get('category')
                    category = Category.objects.filter(name=category_name).first() if category_name else None

                    Product.objects.create(
                        seller=seller,
                        name=row.get('name'),
                        description=row.get('description'),
                        price=Decimal(row.get('price') or '0'),
                        category=category,
                        image=None,
                        approved=False
                    )
                    created_count += 1
                except User.DoesNotExist:
                    skipped_count += 1
                except Exception:
                    skipped_count += 1

            messages.success(request, f"Imported {created_count} products. Skipped {skipped_count} rows.")
            return redirect('dashboard_admin')
    else:
        form = ProductImportForm()

    return render(request, 'core/import_products.html', {'form': form})

# For edit request
@login_required
def request_product_edit(request, product_id):
    product = get_object_or_404(Product, id=product_id, seller=request.user)

    # 🚨 Block if product is not approved
    if not product.approved:
        messages.error(request, "You can only request edits after the product has been approved by admin.")
        return redirect('dashboard_seller')

    if request.method == "POST":
        form = ProductForm(request.POST, request.FILES, instance=product)
        if form.is_valid():
            ProductChangeRequest.objects.create(
                seller=request.user,
                product=product,
                request_type='edit',
                new_name=form.cleaned_data.get('name'),
                new_description=form.cleaned_data.get('description'),
                new_price=form.cleaned_data.get('price'),
                new_quantity=form.cleaned_data.get('quantity'),
                new_category=form.cleaned_data.get('category'),
                new_image=form.cleaned_data.get('image')
            )
            messages.success(request, "Edit request submitted to admin.")
            return redirect('dashboard_seller')
    else:
        form = ProductForm(instance=product)
    return render(request, 'core/request_product_edit.html', {'form': form, 'product': product})


# For delete request
@login_required
def request_product_delete(request, product_id):
    product = get_object_or_404(Product, id=product_id, seller=request.user)

    # 🚨 Block if product is not approved
    if not product.approved:
        messages.error(request, "You can only request deletion after the product has been approved by admin.")
        return redirect('dashboard_seller')

    if request.method == "POST":
        ProductChangeRequest.objects.create(
            seller=request.user,
            product=product,
            request_type='delete'
        )
        messages.warning(request, "Delete request sent to admin.")
        return redirect('dashboard_seller')

    return render(request, 'core/request_product_delete.html', {'product': product})


# Admin: view all requests
def admin_change_requests(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    requests = ProductChangeRequest.objects.filter(approved=False).order_by('-created_at')
    return render(request, 'core/admin_change_requests.html', {'requests': requests})



def approve_change_request(request, request_id):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    change_request = get_object_or_404(ProductChangeRequest, id=request_id)

    if change_request.request_type == 'edit':
        product = change_request.product
        if change_request.new_name: product.name = change_request.new_name
        if change_request.new_description: product.description = change_request.new_description
        if change_request.new_price: product.price = change_request.new_price
        if change_request.new_quantity: product.quantity = change_request.new_quantity
        if change_request.new_category: product.category = change_request.new_category
        if change_request.new_image: product.image = change_request.new_image

        product.approved = True
        product.save()

        change_request.approved = True
        change_request.save()
        messages.success(request, "Edit request approved successfully.")

    elif change_request.request_type == 'delete':
        change_request.product.delete()
        change_request.delete()
        messages.success(request, "Delete request approved and product removed.")

    return redirect('dashboard_admin')



def reject_change_request(request, request_id):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    change_request = get_object_or_404(ProductChangeRequest, id=request_id)
    change_request.delete()
    messages.error(request, "Change request rejected.")
    return redirect('dashboard_admin')
