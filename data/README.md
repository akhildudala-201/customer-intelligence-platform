# Olist Brazilian E-Commerce Dataset (Practice Data)

This is a public, multi table dataset from Olist, a Brazilian e-commerce marketplace. We use it as practice data because it has the same shape as our real membership data. There is a central customer entity that links out to transactions, line items, payments, products, and an engagement signal. Interns can prototype the whole pipeline here before touching the real data.

Source is the Olist public dataset (originally from Kaggle, mirrored on GitHub). It covers orders placed between 2016 and 2018. There are 8 tables and roughly 1.4 million rows in total.

## Tables

| File | Rows | What it holds |
|---|---|---|
| olist_customers_dataset.csv | 99,441 | Customers. Each order gets a customer_id, and customer_unique_id ties a person across many orders |
| olist_orders_dataset.csv | 99,441 | Orders with status and timestamps for purchase, approval, delivery, and estimate |
| olist_order_items_dataset.csv | 112,650 | Line items. One row per product in an order, with price and freight |
| olist_order_payments_dataset.csv | 103,886 | Payments per order, with type and installments |
| olist_order_reviews_dataset.csv | 105,759 | Customer reviews and scores, our engagement and satisfaction signal |
| olist_products_dataset.csv | 32,951 | Product catalog with category and dimensions |
| olist_sellers_dataset.csv | 3,095 | Sellers who fulfil the items |
| olist_geolocation_dataset.csv | 1,000,163 | Zip code to latitude and longitude lookup for enrichment |

## How the tables join

```
customers --customer_id--> orders --order_id--> order_items --product_id--> products
                             |                         '--seller_id--> sellers
                             |--order_id--> order_payments
                             '--order_id--> order_reviews

customers and sellers --zip_code_prefix--> geolocation
```

Join keys in plain terms.

- customers to orders on customer_id
- orders to order_items, order_payments, and order_reviews on order_id
- order_items to products on product_id
- order_items to sellers on seller_id
- customers and sellers to geolocation on zip_code_prefix

One thing to watch. customer_id is unique per order, so to track a real person across orders you group by customer_unique_id, not customer_id.

## How this maps to our membership data

| Olist table | Membership equivalent |
|---|---|
| customers | contact and account |
| orders | invoice header, and the anchor for the churn label |
| order_items | invoicedetail |
| order_payments | invoice payment fields |
| order_reviews | engagement signal |
| products | pa_benefit and the product catalog |

## Deriving a churn label

This dataset has no churn column, which is good practice. In our real data the label comes from renewal cycles. Here you build it yourself from purchase behavior.

A simple starting definition. Group orders by customer_unique_id, then look at the gap between their last purchase and the end of the data window. A customer who has not purchased again within a chosen window, for example 180 days, counts as churned. Treat repeat buyers within the window as retained.

Since most Olist customers buy only once, an alternative framing is repeat purchase prediction. Predict whether a first time buyer will ever come back. Either framing exercises the same skills we need for the real label.

## Suggested next steps for interns

1. Load all 8 CSVs into SQLite and confirm the joins resolve.
2. Build a customer level table keyed on customer_unique_id.
3. Derive a churn or repeat purchase label using the window idea above.
4. Engineer features. Recency, frequency, monetary value, average review score, payment type, and product categories bought.
5. Train a first churn model and build simple value and risk segments for campaigns.
