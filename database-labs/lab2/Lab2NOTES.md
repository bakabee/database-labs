-------------------------------------------------------------------
##//-------Lab 02 – Query 1: Data Exploration Notes------//
-------------------------------------------------------------------

** customers Table **

The customers table contains basic customer information including:

            -customer_id (Primary Key)

            -name

            -email

            -city

            -signup_date

This table stores who the customers are and when they registered. Each customer has a unique ID, which is later used to link them to their orders.
---------------------------------------------------------------------
** products Table **

The products table contains product details:

            -product_id (Primary Key)

            -product_name

            -category

            -price

            -stock_qty

This table stores all items available for sale, including their category (Electronics, Books, Furniture), their price, and how many are currently in stock.
---------------------------------------------------------------------
** orders Table **

The orders table contains order-level information:

            -order_id (Primary Key)

            -customer_id (Foreign Key → customers)

            -order_date

            -status

            -total_amount

            -shipped_date

This table shows which customer placed each order and the overall total for that order. The customer_id connects each order to a specific customer.
---------------------------------------------------------------------
Relationship:

Each customer can have multiple orders (one-to-many relationship).
---------------------------------------------------------------------
** order_items Table **

The order_items table contains detailed line items for each order:

            -item_id (Primary Key)

            -order_id (Foreign Key → orders)

            -product_id (Foreign Key → products)

            -quantity

            -unit_price

This table shows which products were included in each order and how many were purchased.
----------------------------------------------------------------------
Relationship:

One order can have multiple order items.

Each order item references one product.

This creates a many-to-many relationship between orders and products (resolved using order_items).
---------------------------------------------------------------------
** Why Are Some shipped_date Values NULL? **

Some shipped_date values are NULL because:

            -The order status is pending

            -The order is still processing

            -The order was cancelled

            -This indicates the order has not yet been shipped or was never shipped.

Therefore, shipped_date is only filled when an order has actually been shipped or delivered.
---------------------------------------------------------------------
**  Overall Data Insight  **

This dataset represents a typical e-commerce system:

            -Customers sign up

            -Customers place orders

            -Orders contain multiple products

            -Orders move through statuses (pending → shipped → delivered)

            -Not all orders are completed (some are cancelled or still processing)

The structure correctly models a real-world online store database.
-------------------------------------------------------------------
##-------Lab 02 – Query 2: Select Specific Column------
-------------------------------------------------------------------
** Version 1 — Oldest to Newest **

SELECT name, city, signup_date
FROM customers
ORDER BY signup_date;

       -Returns only name, city, and signup date.

       -Shows customers from earliest to latest signup. Useful for tracking growth.
---------------------------------------------------------------------
** Version 2 — Newest to Oldest **

SELECT name, city, signup_date
FROM customers
ORDER BY signup_date DESC;

       -Reverses the order: newest customers first.

       -Quick way to identify recent signups.
---------------------------------------------------------------------
## --------- Query 3 — Filter by Status ----------
---------------------------------------------------------------------
** Delivered Orders **

SELECT order_id, customer_id, total_amount, order_date
FROM orders
WHERE status = 'delivered'
ORDER BY order_date DESC;

      -Lists all delivered orders, most recent first.
---------------------------------------------------------------------
** Pending Orders **

SELECT COUNT(*) AS pending_count
FROM orders
WHERE status = 'pending';

     -Counts orders not yet shipped or processed.
---------------------------------------------------------------------
** Cancelled Orders **

SELECT COUNT(*) AS cancelled_count
FROM orders
WHERE status = 'cancelled';

     -Counts orders that were cancelled.

**Insight: Filtering by status helps understand order fulfillment, pending backlog, and cancellations.**
--------------------------------------------------------------------
