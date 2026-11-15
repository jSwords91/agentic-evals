---
name: metrics
description: Brightleaf's agreed definitions of revenue, orders, average order value, gross profit, gross margin, active customers and repeat rate, and how to report them. Load before calculating any figure.
---

# Metric definitions

These are the definitions the commercial team has agreed. Use them exactly, and say which one
you used.

## Rules for every metric

- Count only completed orders: `orders.status = 'completed'`. Cancelled and refunded orders
  are not sales.
- Leave out test accounts: join `customers` and keep `customers.is_test = 0`.
- Use the price paid, `order_items.unit_price`, not `products.list_price`.
- Periods use `orders.ordered_at`. A month is the calendar month. Q3 is 1 July to 30 September.
  "The last N days" means the N whole days before today, ending yesterday.

## Definitions

| Metric | Definition |
| --- | --- |
| Revenue | Sum of `quantity * unit_price` over the order items of qualifying orders |
| Orders | Number of qualifying orders |
| Average order value | Revenue divided by orders |
| Gross profit | Sum of `quantity * (unit_price - products.unit_cost)` |
| Gross margin | Gross profit divided by revenue, as a percentage |
| Active customers | Number of distinct customers with at least one qualifying order in the period |
| Repeat rate | Of the active customers in a period, the percentage with two or more qualifying orders in that same period |

## Reporting

- Money in GBP, rounded to 2 decimal places. Percentages from 0 to 100, rounded to 1 decimal
  place. Round in the SQL, so the recorded answer is the figure you report.
- For a breakdown, select one row per group: the group's label and its figure.
- Nothing here measures customer satisfaction, NPS, website traffic, marketing spend or stock.
  If a question needs one of those, say the data isn't there.
