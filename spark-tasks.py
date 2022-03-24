import psycopg2
import pandas as pd
import pyspark.sql as pyspark
from pyspark.sql import functions as f
from pyspark.sql.types import *
from sqlalchemy import create_engine
from pyspark.sql.window import Window

appName = "PySpark PostgreSQL Example - via psycopg2"
master = "local"

spark = pyspark.SparkSession.builder.master(master).appName(appName).getOrCreate()

engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost/postgres?client_encoding=utf8")
pandas_actor = pd.read_sql('select * from public.actor', engine)
pandas_address = pd.read_sql('select * from public.address', engine)
pandas_category = pd.read_sql('select * from public.category', engine)
pandas_city = pd.read_sql('select * from public.city', engine)
pandas_country = pd.read_sql('select * from public.country', engine)
pandas_customer = pd.read_sql('select * from public.customer', engine)
pandas_film = pd.read_sql('SELECT * FROM public.film ',
                          engine)
pandas_film_actor = pd.read_sql('select * from public.film_actor', engine)
pandas_film_category = pd.read_sql('select * from public.film_category', engine)
pandas_inventory = pd.read_sql('select * from public.inventory', engine)
pandas_language = pd.read_sql('select * from public.language', engine)
pandas_payment = pd.read_sql('select * from public.payment', engine)
pandas_rental = pd.read_sql('select * from public.rental', engine)
pandas_staff = pd.read_sql('SELECT staff_id, first_name, last_name, address_id, email, store_id, active, username, '
                           '"password", last_update FROM public.staff s ', engine)
pandas_store = pd.read_sql('select * from public.store', engine)

# Convert Pandas dataframe to spark DataFrame

film_schema = [
    StructField('film_id', IntegerType(), True),
    StructField('title', StringType(), True),
    StructField('description', StringType(), True),
    StructField('release_year', IntegerType(), True),
    StructField('language_id', IntegerType(), True),
    StructField('original_language_id', IntegerType(), True),
    StructField('rental_duration', IntegerType(), True),
    StructField('rental_rate', FloatType(), True),
    StructField('length', IntegerType(), True),
    StructField('replacement_cost', FloatType(), True),
    StructField('rating', StringType(), True),
    StructField('last_update', TimestampType(), True),
    StructField('special features', StringType(), True),
    StructField('fulltext', StringType(), True)
]
final_stuct_film = StructType(fields=film_schema)

staff_schema = [
    StructField('staff_id', IntegerType(), True),
    StructField('first_name', StringType(), True),
    StructField('last_name', StringType(), True),
    StructField('address_id', IntegerType(), True),
    StructField('email', StringType(), True),
    StructField('store_id', IntegerType(), True),
    StructField('active', BooleanType(), True),
    StructField('username', StringType(), True),
    StructField('password', StringType(), True),
    StructField('last_update', TimestampType(), True)
]
final_stuct_staff = StructType(fields=staff_schema)

df_actor = spark.createDataFrame(pandas_actor)
df_address = spark.createDataFrame(pandas_address)
df_category = spark.createDataFrame(pandas_category)
df_city = spark.createDataFrame(pandas_city)
df_country = spark.createDataFrame(pandas_country)
df_customer = spark.createDataFrame(pandas_customer)
df_film = spark.createDataFrame(pandas_film, schema=final_stuct_film)
df_film_actor = spark.createDataFrame(pandas_film_actor)
df_film_category = spark.createDataFrame(pandas_film_category)
df_inventory = spark.createDataFrame(pandas_inventory)
df_language = spark.createDataFrame(pandas_language)
df_payment = spark.createDataFrame(pandas_payment)
df_rental = spark.createDataFrame(pandas_rental)
df_staff = spark.createDataFrame(pandas_staff, schema=final_stuct_staff)
df_store = spark.createDataFrame(pandas_store)

# Start working with Spark Df

# I do all tasks by analogy from my tasks . All queries are same

#  Find number of films in each category in descending order

df_number_category = df_film_category.join(df_film, df_film.film_id == df_film_category.film_id, 'inner') \
    .join(df_category, df_category.category_id == df_film_category.category_id, 'inner') \
    .groupby(df_category.name) \
    .agg(f.count(df_film.film_id).alias('number')).sort(f.desc("number")).select(df_category.name,'number')

df_number_category.show()

# Find 10 actors which films was rented more in descending order

df_10_actors_most_rented = df_rental.join(df_inventory, df_inventory.inventory_id == df_rental.inventory_id, 'inner') \
    .join(df_film, df_film.film_id == df_inventory.film_id, 'inner') \
    .join(df_film_actor, df_film.film_id == df_film_actor.film_id, 'inner') \
    .join(df_actor, df_film_actor.actor_id == df_actor.actor_id, 'inner') \
    .groupby(df_actor.actor_id, df_actor.first_name, df_actor.last_name) \
    .agg(f.count(df_rental.rental_id).alias('number')) \
    .sort(f.desc('number')) \
    .select(df_actor.first_name, df_actor.last_name, 'number')
df_10_actors_most_rented.show(10)

# Show film category on which spent the most money

df_most_paid_category = df_payment.join(df_rental, df_rental.rental_id == df_payment.rental_id, "inner") \
    .join(df_inventory, df_inventory.inventory_id == df_rental.inventory_id, 'inner') \
    .join(df_film, df_film.film_id == df_inventory.film_id, 'inner') \
    .join(df_film_category, df_film_category.film_id == df_film.film_id, 'inner') \
    .join(df_category, df_category.category_id == df_film_category.category_id, 'inner') \
    .groupby(df_category.category_id, df_category.name) \
    .agg(f.sum(df_payment.amount).alias('sum_money')) \
    .sort(f.desc('sum_money')) \
    .select(df_category.name, 'sum_money')
df_most_paid_category.show(1)

# Show film titles which not in inventory

df_empty_films_inventory = df_film.join(df_inventory, df_inventory.film_id == df_film.film_id, 'left') \
    .filter(df_inventory.inventory_id.isNull()) \
    .select(df_film.title).distinct()
df_empty_films_inventory.show(50)

# Find top3 actors who played in category "Children" most times

df_children_actors = df_film.join(df_film_actor, df_film.film_id == df_film_actor.film_id, 'inner') \
    .join(df_actor, df_actor.actor_id == df_film_actor.actor_id, 'inner') \
    .join(df_film_category, df_film.film_id == df_film_category.film_id, 'inner') \
    .join(df_category, df_film_category.category_id == df_category.category_id, 'inner') \
    .filter(f.upper(df_category.name) == 'Children'.upper()) \
    .groupby(df_actor.actor_id, df_actor.first_name, df_actor.last_name) \
    .agg(f.count(df_actor.actor_id).alias('number')) \
    .sort(f.desc('number')) \
    .select(df_actor.first_name, df_actor.last_name, 'number')

Windowspec = Window.orderBy(f.desc("number"))
df_3_children_actor = df_children_actors.withColumn('rank', f.dense_rank().over(Windowspec)) \
    .select(df_children_actors.first_name, df_children_actors.last_name, df_children_actors.number, 'rank')

df_top3_actors_children = df_3_children_actor.filter(f.col('rank') <= 3) \
    .select(df_3_children_actor.first_name, df_3_children_actor.last_name, df_3_children_actor.number)
df_top3_actors_children.show()

# Show cities with the number of active and inactive customers (active — customer.active = 1). Sort by the number
# of inactive clients in descending order.

df_customer_true = df_customer.join(df_address, df_address.address_id == df_customer.address_id, 'inner') \
    .join(df_city, df_city.city_id == df_address.city_id, 'inner') \
    .filter(df_customer.activebool == True) \
    .groupby(df_city.city_id, df_city.city) \
    .agg(f.count(df_customer.customer_id).alias('number')) \
    .select(df_city.city, 'number')

df_customer_true = df_customer_true.withColumn("status", f.lit('true')).select(df_customer_true.city,'status',
                                                                               df_customer_true.number)

df_customer_false = df_city.join(df_address, df_address.city_id == df_city.city_id, 'left') \
    .join(df_customer, (df_customer.address_id == df_address.address_id) & (df_customer.activebool == False), 'left') \
    .withColumn('status', f.lit('false')) \
    .withColumn('active', f.when(df_customer.activebool.isNull(), 0).otherwise(1)) \
    .groupby(df_city.city_id,df_city.city,'status') \
    .agg(f.sum('active').alias('count')) \
    .select(df_city.city, 'status', 'count').distinct()

union_all_true_false = df_customer_true.unionAll(df_customer_false)

df_active_customers_city = union_all_true_false.sort(union_all_true_false.city,union_all_true_false.status,union_all_true_false.number)\
    .select(union_all_true_false.city,union_all_true_false.status,union_all_true_false.number)
df_active_customers_city.show(100)

#Find category film in eah town with biggest number of rental hours and city name starts from 'A' or have '-' in name

df_rental_hours = df_rental.join(df_inventory, df_inventory.inventory_id == df_rental.inventory_id, 'inner') \
    .join(df_film, df_film.film_id == df_inventory.film_id, 'inner') \
    .join(df_film_category, df_film.film_id == df_film_category.film_id, 'inner') \
    .join(df_category, df_category.category_id == df_film_category.category_id, 'inner') \
    .join(df_customer, df_customer.customer_id == df_rental.customer_id, 'inner') \
    .join(df_address, df_address.address_id == df_customer.address_id, 'inner') \
    .join(df_city, df_city.city_id == df_address.city_id, 'inner') \
    .filter(f.upper(df_city.city).like('a%'.upper()) | f.upper(df_city.city).like('%-%'.upper())) \
    .groupby(df_city.city, df_category.name) \
    .agg(f.sum((f.unix_timestamp(df_rental.return_date) - f.unix_timestamp(df_rental.rental_date)) / 3600).alias(
    'rental_hours')) \
    .select(df_city.city, df_category.name, 'rental_hours')

Windowspec = Window.partitionBy(df_rental_hours.city).orderBy(df_rental_hours.city,f.desc(df_rental_hours.rental_hours))
df_cities_category = df_rental_hours.withColumn('rank', f.rank().over(Windowspec)).filter(f.col('rank') == 1).select(df_rental_hours.city,
                                                                                                df_rental_hours.name,
                                                                                                df_rental_hours.rental_hours)
df_cities_category.show(600)


