

-- Insert sample users
INSERT INTO users (username, password, email, full_name, role) VALUES
('admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyE3H.w8h0O2', 'admin@example.com', 'Administrator', 'admin'),
('john_doe', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyE3H.w8h0O2', 'john@example.com', 'John Doe', 'user'),
('jane_smith', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyE3H.w8h0O2', 'jane@example.com', 'Jane Smith', 'user'),
('bob_wilson', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyE3H.w8h0O2', 'bob@example.com', 'Bob Wilson', 'user'),
('alice_brown', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyE3H.w8h0O2', 'alice@example.com', 'Alice Brown', 'user');

-- Insert sample products
INSERT INTO products (name, description, price, category, stock) VALUES
('Laptop', 'High-performance laptop', 999.99, 'Electronics', 50),
('Smartphone', 'Latest smartphone model', 699.99, 'Electronics', 100),
('Headphones', 'Noise-cancelling headphones', 199.99, 'Electronics', 75),
('Coffee Maker', 'Programmable coffee maker', 79.99, 'Appliances', 30),
('Desk Chair', 'Ergonomic office chair', 249.99, 'Furniture', 20),
('Monitor', '27-inch 4K monitor', 399.99, 'Electronics', 40),
('Keyboard', 'Mechanical keyboard', 129.99, 'Electronics', 60),
('Mouse', 'Wireless gaming mouse', 59.99, 'Electronics', 80),
('Desk Lamp', 'LED desk lamp', 39.99, 'Furniture', 45),
('Backpack', 'Laptop backpack', 49.99, 'Accessories', 90);

-- Insert sample orders
INSERT INTO orders (user_id, product_id, quantity, total_price, status) VALUES
(2, 1, 1, 999.99, 'completed'),
(2, 3, 1, 199.99, 'completed'),
(3, 2, 1, 699.99, 'pending'),
(4, 5, 1, 249.99, 'completed'),
(5, 4, 2, 159.98, 'shipped');