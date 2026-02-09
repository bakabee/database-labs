-- Lab 1: Analytical Queries 
-- Author : Zarghuna FArman

-- Query 1 : Filtering with WHERE clause
-- Task -- Find all boooks rated above 4.0
SELECT title, autjor, rating, category
FROM books_read
WHERE rating > 4.0
ORDER BY rating DESC;

-- Query 2 : Aggregation with GROUP BY
-- Task -- Find the average rating & number of books per category
SELECT category, AVG(rating) AS avg_rating, COUNT(*) AS book_count 
FROM books_read
GROUP BY category
ORDER BY avg_rating DESC;

-- QUERY 3 : Sorting with ORDER BY
-- Task -- List books from longest to shortest by page count
SELECT title , author, pages, rating
FROM books_read 
ORDER BY pages DESC;

-- QUERY 4 : Date manipulation function
-- Task -- Find books you finished in the month of October 2024
SELECT title , author, date_finished
FROM books_read
WHERE TO_CHAR(date_finished, 'YYYY-MM') = '2024-10';

-- QUERY 5 : Multi-condition Query (AND/OR)
-- Task -- Find books that are either Machine learning category OR rated above 4.5
SELECT title, author, category, rating
FROM books_read
WHERE category = 'Machine Learning' OR rating > 4.5
ORDER BY rating DESC;
