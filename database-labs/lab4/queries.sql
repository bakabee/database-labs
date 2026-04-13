-- ============================================================
-- Lab 04: SQL Joins, Aggregations, and Analytics Queries
-- Author: [Your Name]
-- Date: 2026-04-13
-- ============================================================


-- ------------------------------------------------------------
-- Query 1: Orders with Customer Details
-- ------------------------------------------------------------
SELECT o.order_id,
       c.name AS customer_name,
       c.city,
       o.order_date,
       o.status,
       o.total_amount
FROM orders o
INNER JOIN customers c 
    ON o.customer_id = c.customer_id
ORDER BY o.order_date DESC;


-- ------------------------------------------------------------
-- Query 2: Full Order Receipt (4-table join)
-- ------------------------------------------------------------
SELECT c.name AS customer,
       o.order_id,
       o.order_date,
       o.status,
       p.product_name,
       p.category,
       oi.quantity,
       oi.unit_price,
       oi.quantity * oi.unit_price AS line_total
FROM order_items oi
INNER JOIN orders o 
    ON oi.order_id = o.order_id
INNER JOIN customers c 
    ON o.customer_id = c.customer_id
INNER JOIN products p 
    ON oi.product_id = p.product_id
ORDER BY o.order_id, p.category;


-- ------------------------------------------------------------
-- Query 3: Customers Who Never Ordered
-- ------------------------------------------------------------
SELECT c.customer_id,
       c.name,
       c.email,
       c.city,
       c.signup_date
FROM customers c
LEFT JOIN orders o 
    ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL
ORDER BY c.signup_date;


-- ------------------------------------------------------------
-- Query 4: Products Never Ordered
-- ------------------------------------------------------------
SELECT p.product_id,
       p.product_name,
       p.category,
       p.price,
       p.stock_qty
FROM products p
LEFT JOIN order_items oi 
    ON p.product_id = oi.product_id
WHERE oi.item_id IS NULL
ORDER BY p.category;


-- ------------------------------------------------------------
-- Query 5: Revenue by Customer
-- ------------------------------------------------------------
SELECT c.name,
       c.city,
       COUNT(o.order_id) AS total_orders,
       COALESCE(SUM(o.total_amount), 0) AS total_spent
FROM customers c
LEFT JOIN orders o 
    ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.city
ORDER BY total_spent DESC;


-- ------------------------------------------------------------
-- Query 6: Best-Selling Products
-- ------------------------------------------------------------
SELECT p.product_name,
       p.category,
       p.price,
       SUM(oi.quantity) AS units_sold,
       SUM(oi.quantity * oi.unit_price) AS total_revenue
FROM products p
INNER JOIN order_items oi 
    ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.category, p.price
ORDER BY units_sold DESC
LIMIT 5;


-- ------------------------------------------------------------
-- Query 7: City-Level Sales Report
-- ------------------------------------------------------------
SELECT c.city,
       COUNT(DISTINCT c.customer_id) AS num_customers,
       COUNT(DISTINCT o.order_id) AS num_orders,
       SUM(o.total_amount) AS total_revenue,
       ROUND(AVG(o.total_amount), 2) AS avg_order_value
FROM customers c
INNER JOIN orders o 
    ON c.customer_id = o.customer_id
GROUP BY c.city
ORDER BY total_revenue DESC;


-- ------------------------------------------------------------
-- Query 8: Delivered Orders (2025 Customer Analysis)
-- ------------------------------------------------------------
SELECT c.name AS customer,
       c.city,
       COUNT(DISTINCT p.product_id) AS distinct_products_bought,
       SUM(oi.quantity) AS total_items,
       SUM(oi.quantity * oi.unit_price) AS total_spent
FROM order_items oi
INNER JOIN orders o 
    ON oi.order_id = o.order_id
INNER JOIN customers c 
    ON o.customer_id = c.customer_id
INNER JOIN products p 
    ON oi.product_id = p.product_id
WHERE o.status = 'delivered'
  AND o.order_date >= '2025-01-01'
GROUP BY c.customer_id, c.name, c.city
ORDER BY total_spent DESC;


-- -----------------------------------------------------------
-- Query 9: Session + Purchase Audit (Data Quality)
-- ------------------------------------------------------------
SELECT c.customer_id AS customer_table_id,
       c.name,
       s.customer_id AS session_table_cust_id,
       COUNT(s.session_id) AS session_count,
       CASE
           WHEN c.customer_id IS NULL THEN 'Orphaned session'
           WHEN s.customer_id IS NULL THEN 'Customer with no sessions'
           ELSE 'Matched'
       END AS audit_status
FROM customers c
FULL OUTER JOIN user_sessions s 
    ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.name, s.customer_id
ORDER BY audit_status, c.name;
------------------------------------------------------------
-- Query 10: Session vs Purchase Behaviour (Customer Funnel)
------------------------------------------------------------

SELECT c.name,
       c.city,
       COUNT(DISTINCT s.session_id) AS total_sessions,
       COUNT(DISTINCT o.order_id) AS total_orders,
       COALESCE(SUM(o.total_amount), 0) AS total_spent,
       CASE
           WHEN COUNT(DISTINCT o.order_id) > 0 THEN 'Converted'
           ELSE 'Not Converted'
       END AS customer_status
FROM customers c
INNER JOIN user_sessions s 
    ON c.customer_id = s.customer_id
LEFT JOIN orders o 
    ON c.customer_id = o.customer_id
    AND o.status = 'delivered'
GROUP BY c.customer_id, c.name, c.city
ORDER BY total_sessions DESC;
