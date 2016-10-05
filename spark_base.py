from pyspark.sql import SparkSession
from pyspark.sql.functions import log, col, udf
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.types import DoubleType

import numpy as np

spark = SparkSession \
            .builder \
            .master('local') \
            .appName('challenge25') \
            .config("spark.sql.warehouse.dir", "file:///D:/tools/Anaconda2/Scripts/spark-warehouse") \
            .getOrCreate()
            #https://issues.apache.org/jira/browse/SPARK-15893
            
try:
    traindf = spark.read.csv('D:/Dev/dataScience/kaggle/challenge_25_data/boites_medicaments_train.csv', sep=';', header=True)
    testdf = spark.read.csv('D:/Dev/dataScience/kaggle/challenge_25_data/boites_medicaments_test.csv', sep=';', header=True)
        
    traindf = traindf.withColumn('logprix', log('prix'))
    
    # features numériques
    feat_num = ['libelle_plaquette', 'libelle_ampoule', 'libelle_flacon', 
                'libelle_tube', 'libelle_stylo', 'libelle_seringue',
                'libelle_pilulier', 'libelle_sachet', 'libelle_comprime', 
                'libelle_gelule', 'libelle_film', 'libelle_poche',
                'libelle_capsule'] + ['nb_plaquette', 'nb_ampoule', 
                'nb_flacon', 'nb_tube', 'nb_stylo', 'nb_seringue',
                'nb_pilulier', 'nb_sachet', 'nb_comprime', 'nb_gelule', 
                'nb_film', 'nb_poche', 'nb_capsule', 'nb_ml']
    # features date
    feat_dates = ['date declar annee', 'date amm annee']
    # features catégorielles
    feat_cat = ['statut', 'etat commerc', 'agrement col', 'tx rembours',
              'voies admin', 'statut admin', 'type proc']
             
    # features texte
    feat_text = ['libelle', 'titulaires', 'substances', 'forme pharma']
    
    # Encode text categories to numeri values
    feat_cat_idx = []
    for cat in feat_cat:    
        indexedCat = cat + '_idx'
        feat_cat_idx.append(indexedCat)
        indexer = StringIndexer(inputCol=cat, outputCol=indexedCat)
        traindf = indexer.fit(traindf).transform(traindf).drop(cat)
        testdf = indexer.fit(testdf).transform(testdf).drop(cat)
    
    # Define features used for estimation
    features = feat_num + feat_cat_idx
        
    # Convert all features to Double
    expr = [col(c).cast(DoubleType()).alias(c) for c in features]
    traindf_temp = traindf.select(col('logprix'), *expr)
    testdf_temp = testdf.select(*expr)

    # Assemble all features in a single column (name:'features', type:Vector)
    vectorAssembler = VectorAssembler(inputCols=features, outputCol='features')
    train_lp_df = vectorAssembler.transform(traindf_temp).select(col('logprix'), col('features'))
    
    # Split dataset for train & test
    (train, test) = train_lp_df.randomSplit([0.8, 0.2])
    
    # Define estimator 
    rf = RandomForestRegressor(labelCol='logprix', 
                               featuresCol='features',
                               numTrees=10, 
                               featureSubsetStrategy='auto', 
                               maxDepth=30)
    
    #evaluator = RegressionEvaluator(metricName="mae")
    #paramGrid = ParamGridBuilder().build()
    #crossval = CrossValidator(estimator=rf,
    #                          estimatorParamMaps=paramGrid,
    #                          evaluator=evaluator,
    #                          numFolds=5)
    #model = crossval.fit(train)
    
    # Train model  
    model = rf.fit(train)
    
    # Evaluate model 
    predictions = model.transform(test)
    
    # Mean Absolute Percentage Error
    def mape_error(log_y_true, log_y_pred): 
        y_true = np.exp(log_y_true)
        y_pred = np.exp(log_y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Compute test error
    error = predictions.rdd.map(lambda row: mape_error(row['logprix'], row['prediction'])).mean()
    print error
    

finally:
    spark.stop()