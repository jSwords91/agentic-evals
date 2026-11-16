"""Brightleaf's sales warehouse: the schema, a description of every table and column, and data.

Brightleaf is a fictional UK online shop selling houseplants, pots and plant care. In production
this would be your real warehouse (Postgres, BigQuery, Snowflake...) and the descriptions would
come from your data catalog or dbt docs. Here it's SQLite, generated from a fixed seed so every
trial sees exactly the same rows.

The data has the traps real warehouses have, because they're what make an analyst agent's
numbers wrong:
  - test accounts, whose orders must be left out of every metric
  - cancelled and refunded orders, which aren't sales
  - a price paid per item (order_items.unit_price) that isn't always the list price
  - this morning's orders: today isn't over, so a period that includes it is incomplete
"""

from __future__ import annotations

import random
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, region TEXT NOT NULL,
    signup_date TEXT NOT NULL, is_test INTEGER NOT NULL DEFAULT 0);
CREATE TABLE products (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
    unit_cost REAL NOT NULL, list_price REAL NOT NULL);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers (id),
    ordered_at TEXT NOT NULL, channel TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders (id), product_id INTEGER NOT NULL REFERENCES products (id),
    quantity INTEGER NOT NULL, unit_price REAL NOT NULL);
"""

# What the agent reads about each table and column, through describe_table. Good descriptions
#   are the difference between a right join and a plausible wrong one: say what a row is, how
#   tables join, and every value a coded column can take.
TABLES = {
    "customers": "One row per customer account.",
    "products": "One row per product in the catalogue.",
    "orders": "One row per order, whatever its status.",
    "order_items": "One row per product line in an order. Join to orders on order_id and to products on product_id.",
}
COLUMNS = {
    "customers": {
        "id": "Customer id. orders.customer_id refers to it.",
        "name": "Customer's name.",
        "region": "UK region: North, Midlands, South or Scotland.",
        "signup_date": "Date the account was created, YYYY-MM-DD.",
        "is_test": "1 for internal test accounts, 0 for real customers.",
    },
    "products": {
        "id": "Product id. order_items.product_id refers to it.",
        "name": "Product name.",
        "category": "Houseplants, Pots or Care.",
        "unit_cost": "What one unit costs Brightleaf, in GBP.",
        "list_price": "Current catalogue price in GBP. Not always what was paid: see order_items.unit_price.",
    },
    "orders": {
        "id": "Order id.",
        "customer_id": "The customer who placed the order. Joins to customers.id.",
        "ordered_at": "When the order was placed, UK time, YYYY-MM-DD HH:MM:SS.",
        "channel": "Where the order was placed: web or app.",
        "status": "completed, cancelled or refunded.",
    },
    "order_items": {
        "order_id": "Joins to orders.id.",
        "product_id": "Joins to products.id.",
        "quantity": "Units of the product in this order.",
        "unit_price": "Price paid per unit in GBP, after any discount.",
    },
}

REGIONS = ["North", "Midlands", "South", "Scotland"]
PRODUCTS = [  # id, name, category, unit_cost, list_price
    (1, "Monstera deliciosa", "Houseplants", 9.50, 24.00),
    (2, "Snake plant", "Houseplants", 6.00, 16.00),
    (3, "Fiddle leaf fig", "Houseplants", 14.00, 38.00),
    (4, "Pothos", "Houseplants", 3.50, 11.00),
    (5, "Peace lily", "Houseplants", 5.50, 15.00),
    (6, "Olive tree", "Houseplants", 22.00, 55.00),
    (7, "Terracotta pot, small", "Pots", 2.00, 6.50),
    (8, "Terracotta pot, large", "Pots", 5.00, 14.00),
    (9, "Glazed pot", "Pots", 8.00, 22.00),
    (10, "Houseplant feed", "Care", 2.20, 7.50),
    (11, "Repotting mix", "Care", 3.00, 9.00),
    (12, "Mister", "Care", 2.50, 8.00),
]
FIRST_DAY, LAST_DAY = date(2026, 7, 1), date(2026, 11, 15)  # "today" is 16 November 2026
TEST_ACCOUNTS = 3
# Orders placed so far today, the morning of Monday 16 November: customer id, time, channel, and
#   (product id, quantity) lines at list price. Fixed rather than drawn from the seed, so every
#   earlier row stays exactly as it was.
TODAY = [
    (12, "07:14", "app", [(1, 1), (9, 1)]),
    (47, "07:52", "web", [(6, 1)]),
    (88, "08:20", "app", [(4, 2)]),
]


def rows(seed: int = 2026) -> tuple[list, list, list]:
    """Customers, orders and order items. The same seed always gives the same rows."""
    rng = random.Random(seed)
    customers = [(i, f"Customer {i}", rng.choices(REGIONS, [30, 25, 35, 10])[0],
                  (date(2025, 1, 1) + timedelta(days=rng.randrange(540))).isoformat(), 0)
                 for i in range(1, 151)]
    customers += [(150 + i, f"QA test account {i}", "South", "2025-01-01", 1) for i in range(1, TEST_ACCOUNTS + 1)]

    orders, items, day = [], [], FIRST_DAY
    while day <= LAST_DAY:
        for _ in range(rng.randint(2, 6)):
            customer = rng.choice([c for c in customers[:150] if c[3] <= day.isoformat()])
            order_id = len(orders) + 1
            ordered_at = f"{day} {rng.randint(7, 22):02d}:{rng.randint(0, 59):02d}:00"
            status = rng.choices(["completed", "cancelled", "refunded"], [86, 8, 6])[0]
            orders.append((order_id, customer[0], ordered_at, rng.choices(["web", "app"], [60, 40])[0], status))
            discount = 0.9 if rng.random() < 0.15 else 1.0
            for product in rng.sample(PRODUCTS, rng.choices([1, 2, 3], [55, 30, 15])[0]):
                items.append((order_id, product[0], rng.choices([1, 2, 3], [70, 22, 8])[0],
                              round(product[4] * discount, 2)))
        if day.day == 10:  # test accounts place a large order every month
            order_id = len(orders) + 1
            orders.append((order_id, 150 + rng.randint(1, TEST_ACCOUNTS), f"{day} 12:00:00", "web", "completed"))
            items.append((order_id, 6, 20, 55.00))
        day += timedelta(days=1)

    prices = {p[0]: p[4] for p in PRODUCTS}
    for customer_id, time, channel, lines in TODAY:
        order_id = len(orders) + 1
        orders.append((order_id, customer_id, f"{LAST_DAY + timedelta(days=1)} {time}:00", channel, "completed"))
        items += [(order_id, product_id, quantity, prices[product_id]) for product_id, quantity in lines]
    return customers, orders, items


def create_db(path: str | Path) -> Path:
    """A fresh copy of the warehouse. Every eval trial gets its own (E8)."""
    path = Path(path)
    customers, orders, items = rows()
    with closing(sqlite3.connect(path)) as con, con:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)
        con.executemany("INSERT INTO products VALUES (?,?,?,?,?)", PRODUCTS)
        con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
        con.executemany("INSERT INTO order_items VALUES (?,?,?,?)", items)
    return path
