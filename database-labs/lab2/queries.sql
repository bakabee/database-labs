---------------------------------------------------------------------
------------- Query 1 – Explore the Data ----------------------------
---------------------------------------------------------------------
--1. Customers Table
SQL Code : SELECT * FROM customers LIMIT 5;
* Explanation *
The first few rows show that the customer base is spread across major cities like Peshawar, Lahore, Karachi, Islamabad. Signup dates range from March 2023 to January 2025, indicating continuous growth of the user base. Customers such as Ali Hassan and Sara Khan represent early signups. This shows that the platform is actively acquiring users across multiple cities over time.

--2. Products Table
SQL Code : SELECT * FROM products LIMIT 5;
* Explanation *
The results show that the products table contains product details such as name, category, price, and available stock quantity. The data indicates that products are grouped into categories in which only electronics are listed here. Stock quantities also vary widely.

--3. Order Table
SQL Code : SELECT * FROM orders LIMIT 5;
* Explanation *
The output shows that each order is linked to a customer through customer_id, confirming a one-to-many relationship between customers and orders. The table also includes order status and shipped dates, indicating that some orders move through different fulfillment stages. Most orders are delivered.

--4. Order Item Table
SQL Code : SELECT * FROM order_items LIMIT 5;
Explanation
The results show that each order can contain multiple products, as seen through repeated order_id values with different product_ids. This table acts as a bridge between orders and products, resolving the many-to-many relationship between them. e.g item 2 contains 2 orders with different products.
----------------------------------------------------------------------
-------------- Query 2 –  Select specific columns--------------------
--------------------------------------------------------------------- 
---Version 1 – Oldest to Newest (Ascending Order)
SQL Code : 
SELECT name, city, signup_date
FROM customers
ORDER BY signup_date;
Explanation :
The results show customers ordered from the earliest signup date to the most recent. This reveals that customer registrations have steadily increased over time, with newer customers appearing at the bottom of the list. e.g Sorting customers by signup_date shows that the earliest users (Ali Hassan, Sara Khan) registered in 2023, while newer customers like Kiran Baig signed up in 2025.

---Version 2 – Newest to Oldest (Descending Order)
SQL Code : 
SELECT name, city, signup_date
FROM customers
ORDER BY signup_date DESC;
When ordered in descending order, the newest customers appear first. This makes it easier to quickly identify the most recently registered users, showing that customer signups continued into 2025. e.g Customers like Kiran Baig, Asad Nawaz, and Maira Aslam represent the newest users.
----------------------------------------------------------------------
--------------- Query 3 – Filter by status --------------------------
---------------------------------------------------------------------
By filtering on status, we get a sense of how many orders are delivered, pending, or cancelled. This is a key step in early-stage data exploration for operational reporting.
--1. Delivered Orders
SQL Code :
SELECT order_id, customer_id, total_amount, order_date
FROM orders
WHERE status = 'delivered'
ORDER BY order_date DESC;
FROM orders
WHERE status = 'delivered';
 ** Explanation :This query shows all orders that have been successfully completed and shipped to customers. By ordering the results by order_date descending, the most recent deliveries appear first, allowing us to quickly identify the latest fulfilled orders. This reveals which customers received their orders and how frequently orders are being delivered over time.
--2. Pending Orders :
SQL Code :
SELECT order_id, customer_id, total_amount, order_date
FROM orders
WHERE status = 'pending’
ORDER BY order_date DESC;
SELECT COUNT(*) AS pending_count
FROM orders
WHERE status = 'pending';
** Explanation: Orders 16, 20, and 14 are pending, indicating they have been placed but not yet shipped. These orders are still active, which is important for operational tracking.
-- 3. Cancelled Orders :
SQL Code :
SELECT order_id, customer_id, total_amount, order_date
FROM orders
WHERE status = 'delivered'
ORDER BY order_date DESC;
SELECT COUNT(*) AS cancelled_count
FROM orders
WHERE status = 'cancelled';
** Explanation :
Only order 12 is cancelled. This is minimal, suggesting customers rarely cancel, or the platform effectively manages orders to reduce cancellations.
---------------------------------------------------------------------
---------------Query 4 – Filter Products by Price--------------------
---------------------------------------------------------------------
-- Method A – Using BETWEEN
SQL Code : 
SELECT product_name, category, price
FROM products
WHERE price BETWEEN 1000 AND 5000
ORDER BY price;
-- Method B – Using Comparison Operators
SQL Code : 
SELECT product_name, category, price
FROM products
WHERE price >= 1000 AND price <= 5000
ORDER BY price;
** Explanation :
This demonstrates that the majority of the product catalog is concentrated in the affordable to mid-price range, catering to everyday consumer needs. It also shows that Books are consistently low-cost, while Electronics span both mid and high ranges, indicating a strategy to balance accessibility with premium offerings.
---------------------------------------------------------------------
--------------Query 5 – Top 10 highest- value orders-----------------
---------------------------------------------------------------------
SQL Code : 
SELECT order_id, customer_id, total_amount, status
FROM orders
ORDER BY total_amount DESC
LIMIT 10;
** Explanation :
The results show that the highest-value orders range from very large purchases such as 43,000 to lower but still significant amounts within the top 10. These high-value orders are spread across different customers and are not limited to a single status — some are marked as delivered, while one as pending. This indicates that expensive purchases are common in the system and that not all high-value transactions are immediately completed, which makes them important for monitoring, operational control, and potential fraud review.
--------------------------------------------------------------------
-----------------Query 6 – Multi-condition Filter--------------
--------------------------------------------------------------------
SQL Code : 
SELECT order_id, customer_id, total_amount, order_date
FROM orders
WHERE status = 'delivered'
AND order_date >= '2025-01-01'
AND total_amount > 10000
ORDER BY total_amount DESC;
** Explanation :
The results show only high-value orders from 2025 that have already been successfully delivered. This means smaller purchases, orders from previous years, and orders that are pending or cancelled are excluded. The output highlights customers who made significant purchases this year, which could indicate strong repeat buyers or demand for premium products in 2025. Sorting by total amount makes it easy to identify the highest-spending customers first.
---------------------------------------------------------------------
---------------Query 7 – Pattern matching on email----------------
---------------------------------------------------------------------
SQL Code :
SELECT name, email, city
FROM customers
WHERE email LIKE '%@gmail.com'
ORDER BY name;
** Explanation :
- The results show all customers whose email addresses end with @gmail.com, meaning they use Gmail as their email provider. This indicates that Gmail is a commonly used provider among customers in the dataset. 
- When replacing LIKE with ILIKE, the results remain the same because all stored email addresses are already in lowercase, so case sensitivity does not affect the outcome.
- Using LIKE '%yahoo%' returns customers who use Yahoo email accounts, showing a smaller group compared to Gmail users.
- Finally, LIKE 'a%' returns customers whose names start with the letter “a”, helping identify users alphabetically and demonstrating how pattern matching can filter data based on prefixes.
---------------------------------------------------------------------
------------Query 8 – NULL handling (unshipped orders)-----------
---------------------------------------------------------------------
SQL Code :
SELECT order_id, customer_id, order_date, status, total_amount
FROM orders
WHERE shipped_date IS NULL
ORDER BY order_date;
** Explanation  :
The results show all orders that have not yet been shipped. These typically include orders with statuses such as pending, processing, or cancelled, confirming that a missing shipped_date indicates the order has not been completed or dispatched. This helps identify active or incomplete transactions that may require operational attention. 
This query returns zero rows because NULL does not behave like a normal value.
---------------------------------------------------------------------
-------------------Query 9 – Computed column-------------------------
---------------------------------------------------------------------
SQL Code :
SELECT product_name,
category,
price AS original_price,
ROUND(price * 0.80, 2) AS discounted_price,
ROUND(price * 0.20, 2) AS you_save
FROM products
ORDER BY discounted_price DESC;
** Explanation :
The results display each product along with its original price, the new price after a 20% discount, and the exact amount saved in rupees. Higher-priced items show much larger savings compared to lower-cost books, demonstrating how percentage-based discounts scale with price.
---------------------------------------------------------------------
--------------Query 10 – Bring everything together-------------------
---------------------------------------------------------------------
SQL Code : 
SELECT order_id,
customer_id,
total_amount,
order_date,
status,
CASE
WHEN total_amount > 10000 THEN 'URGENT'
ELSE 'NORMAL'
END AS priority
FROM orders
WHERE shipped_date IS NULL
AND order_date >= '2025-01-01'
ORDER BY total_amount DESC
LIMIT 5;
** Explanation :
The results show the five highest-value orders from 2025 that have not yet been shipped. These are active or incomplete transactions and represent the most financially significant pending orders. Orders above 10,000 are marked as URGENT, helping quickly identify which unshipped orders require immediate attention. 
-- Extended Version — Three Priority Levels
SQL Code : 
SELECT order_id,
customer_id,
total_amount,
order_date,
status,
CASE
WHEN total_amount > 20000 THEN 'CRITICAL'
WHEN total_amount BETWEEN 5000 AND 20000 THEN 'URGENT'
ELSE 'NORMAL'
END AS priority
FROM orders
WHERE shipped_date IS NULL
AND order_date >= '2025-01-01'
ORDER BY total_amount DESC
LIMIT 5;
** Explanation : 
With three priority levels, the output provides a clearer risk classification. Orders above 20,000 are labeled CRITICAL, identifying extremely high-value pending transactions. Orders between 5,000 and 20,000 are marked URGENT, while smaller orders are categorized as NORMAL. This tiered system improves operational decision-making by allowing teams to focus first on the most financially impactful unshipped orders.
---------------------------------------------------------------------
