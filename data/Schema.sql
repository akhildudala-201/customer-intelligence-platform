
CREATE DATABASE IF NOT EXISTS olist
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE olist;

CREATE TABLE IF NOT EXISTS customers (
    customer_id              CHAR(32)    NOT NULL,
    customer_unique_id       CHAR(32)    NOT NULL,
    customer_zip_code_prefix CHAR(5)     NOT NULL,
    customer_city            VARCHAR(64) NOT NULL,
    customer_state           CHAR(2)     NOT NULL,

    PRIMARY KEY (customer_id),

    INDEX idx_customers_unique_id (customer_unique_id),
    INDEX idx_customers_zip (customer_zip_code_prefix)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
  
  
  
  CREATE TABLE IF NOT EXISTS geolocation (
    id                          INT UNSIGNED NOT NULL AUTO_INCREMENT,
    geolocation_zip_code_prefix CHAR(5)      NOT NULL,
    geolocation_lat             DECIMAL(17,14) NOT NULL,
    geolocation_lng             DECIMAL(17,14) NOT NULL,
    geolocation_city            VARCHAR(64)  NOT NULL,
    geolocation_state           CHAR(2)      NOT NULL,

    PRIMARY KEY (id),

    INDEX idx_geo_zip (geolocation_zip_code_prefix)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
  
  
  
  CREATE TABLE IF NOT EXISTS sellers (
    seller_id              CHAR(32)    NOT NULL,
    seller_zip_code_prefix CHAR(5)     NOT NULL,
    seller_city            VARCHAR(64) NOT NULL,
    seller_state           CHAR(2)     NOT NULL,

    PRIMARY KEY (seller_id),

    INDEX idx_sellers_zip (seller_zip_code_prefix)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
  
  
  
  CREATE TABLE IF NOT EXISTS products (
    product_id                  CHAR(32)    NOT NULL,
    product_category_name       VARCHAR(64) NULL,
    product_name_lenght         SMALLINT   NULL,
    product_description_lenght  SMALLINT   NULL,
    product_photos_qty          SMALLINT   NULL,
    product_weight_g            INT        NULL,
    product_length_cm           INT        NULL,
    product_height_cm           INT        NULL,
    product_width_cm            INT        NULL,

    PRIMARY KEY (product_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
  
  
  CREATE TABLE IF NOT EXISTS orders (
    order_id                      CHAR(32)    NOT NULL,
    customer_id                   CHAR(32)    NOT NULL,
    order_status                  VARCHAR(16) NOT NULL,
    order_purchase_timestamp      DATETIME    NOT NULL,
    order_approved_at             DATETIME    NULL,
    order_delivered_carrier_date  DATETIME    NULL,
    order_delivered_customer_date DATETIME    NULL,
    order_estimated_delivery_date DATETIME    NOT NULL,

    PRIMARY KEY (order_id),

    INDEX idx_orders_customer (customer_id),

    CONSTRAINT fk_orders_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
  
  
  
  CREATE TABLE IF NOT EXISTS order_items (
    order_id            CHAR(32)      NOT NULL,
    order_item_id       SMALLINT      NOT NULL,
    product_id          CHAR(32)      NOT NULL,
    seller_id           CHAR(32)      NOT NULL,
    shipping_limit_date DATETIME      NOT NULL,
    price               DECIMAL(10,2) NOT NULL,
    freight_value       DECIMAL(10,2) NOT NULL,

    PRIMARY KEY (order_id, order_item_id),

    INDEX idx_items_product (product_id),
    INDEX idx_items_seller (seller_id),

    CONSTRAINT fk_items_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id),

    CONSTRAINT fk_items_product
        FOREIGN KEY (product_id)
        REFERENCES products(product_id),

    CONSTRAINT fk_items_seller
        FOREIGN KEY (seller_id)
        REFERENCES sellers(seller_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
  
  
  
  CREATE TABLE IF NOT EXISTS order_payments (
    order_id             CHAR(32)      NOT NULL,
    payment_sequential   SMALLINT      NOT NULL,
    payment_type         VARCHAR(16)   NOT NULL,
    payment_installments SMALLINT      NOT NULL,
    payment_value        DECIMAL(10,2) NOT NULL,

    PRIMARY KEY (order_id, payment_sequential),

    CONSTRAINT fk_payments_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
  
  
  
  CREATE TABLE IF NOT EXISTS order_reviews (
    id                       INT UNSIGNED NOT NULL AUTO_INCREMENT,
    review_id                CHAR(32)     NOT NULL,
    order_id                 CHAR(32)     NOT NULL,
    review_score             TINYINT      NOT NULL,
    review_comment_title     VARCHAR(128) NULL,
    review_comment_message   TEXT         NULL,
    review_creation_date     DATETIME     NOT NULL,
    review_answer_timestamp  DATETIME     NOT NULL,

    PRIMARY KEY (id),

    INDEX idx_reviews_review_id (review_id),
    INDEX idx_reviews_order (order_id),

    CONSTRAINT fk_reviews_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


