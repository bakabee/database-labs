-- Lab 1: Books Read Tracker
-- Author: Zarghuna Farman
-- Date: 5th February, 2026

--Create table
CREATE TABLE books_read (
        book-id SERIAL PRIMARY KEY, 
        title VARCHAR(200) NOT NULL,
        author VARCHAR(100) NOT NULL,
        category VARCHAR(50),
        pages INTEGER CHECK (pages > 0),
        date_finished DATE,
        rating DECIMAL(3,1) CHECK (rating >=0 AND rating <=5.0),
        notes TEXT
  );

--Insert sample data
INSERT INTO books_read (title, author, category, pages, date_finished, rating, notes) 
VALUES [paste your INSERT statements];
