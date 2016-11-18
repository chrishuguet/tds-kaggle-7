from pyspark.sql import SparkSession
from pyspark.sql.functions import log, col, udf, split, explode, lit, trim
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.types import DoubleType, IntegerType
import time


import numpy as np

def logTime(out):
    global t0
    print out + ':' + str(time.time() - t0)
    t0 = time.time()
    
def getDistinctValues(df, col_name):
    # type: (DataFrame, str) -> *str
    distinct_values_rows = (df
                      .select(split(col(col_name), ',').alias(col_name))
                      .select(explode(col(col_name)).alias(col_name))
                      .select(trim(col(col_name)))
                      .distinct()
                      .collect())
    return [ row[0] for row in distinct_values_rows ]

def expandOHE(df, categ_name, categ_values):
    # type: (DataFrame, str, *str) -> DataFrame
    func_udf = udf(lambda val, lst: 1 if 'oral' in lst else 0, IntegerType())
    for categ_value in categ_values:
        print 'categ_value' + categ_value
        print 'categ_name' + categ_name
        #func_udf = udf(lambda values: 1 if categ_value in values else 0, IntegerType())
        df = df.withColumn(categ_value, func_udf(lit(categ_value), col(categ_name)))
    logTime('drop')
    return df.drop(col(categ_name))   
        
def nValuedColToBinaryColsOHE(maindf, otherdf, col_name):
    # Get dinstinct col values from main df only
    logTime('getDistinctValues')
    distinct_values = getDistinctValues(maindf, col_name)
    # Expand
    print str(len(distinct_values))
    logTime('expandOHE main')
    maindf_res = expandOHE(maindf, col_name, distinct_values)
    logTime('expandOHE other')
    otherdf_res = expandOHE(otherdf, col_name, distinct_values)
    return maindf_res, otherdf_res, distinct_values    

t0 = 0
logTime('start')
spark = SparkSession \
            .builder \
            .master('local') \
            .appName('challenge25') \
            .config('spark.sql.warehouse.dir', 'file:///D:/tools/Anaconda2/Scripts/spark-warehouse') \
            .getOrCreate()
            # Conf "spark.sql.warehouse.dir" --> https://issues.apache.org/jira/browse/SPARK-15893

spark.sparkContext.setCheckpointDir('file:///D:/Dev/dataScience/kaggle/challenge_25_data/cache')
            
try:
    logTime('read')

    traindf = spark.read.csv('D:/Dev/dataScience/kaggle/challenge_25_data/full/boites_medicaments_train.csv', sep=';', header=True)
    testdf = spark.read.csv('D:/Dev/dataScience/kaggle/challenge_25_data/full/boites_medicaments_test.csv', sep=';', header=True)
    
    logTime('logprix')
    t0 = time.time()    
    traindf = traindf.withColumn('logprix', log('prix'))
    
    # features numériques
    feat_num = ['libelle_plaquette', 'libelle_ampoule', 'libelle_flacon', 'libelle_tube', 'libelle_stylo', 'libelle_seringue',
                'libelle_pilulier', 'libelle_sachet', 'libelle_comprime', 'libelle_gelule', 'libelle_film', 'libelle_poche',
                'libelle_capsule'] + ['nb_plaquette', 'nb_ampoule', 'nb_flacon', 'nb_tube', 'nb_stylo', 'nb_seringue',
                'nb_pilulier', 'nb_sachet', 'nb_comprime', 'nb_gelule', 'nb_film', 'nb_poche', 'nb_capsule', 'nb_ml']
    # features date
    feat_dates = ['date declar annee', 'date amm annee']
    # features catégorielles
    feat_cat = ['statut', 'etat commerc', 'agrement col', 'tx rembours', 'statut admin', 'type proc']
    # features texte
    feat_text = ['libelle', 'titulaires', 'substances', 'forme pharma']
    
    # Encode text categories to numeric values
    logTime('feat_cat')
    feat_cat_idx = []
    for cat in feat_cat:    
        indexedCat = cat + '_idx'
        feat_cat_idx.append(indexedCat)
        indexer = StringIndexer(inputCol=cat, outputCol=indexedCat)
        traindf = indexer.fit(traindf).transform(traindf)
        testdf = indexer.fit(testdf).transform(testdf)
        
    # Transform multi-valued categories to binary values, creating 1 column per category label (ohe style)
    logTime('substances')
    traindf, testdf, feat_substances_ohe = nValuedColToBinaryColsOHE(traindf, testdf, 'voies admin')
    traindf, testdf, feat_substances_ohe = nValuedColToBinaryColsOHE(traindf, testdf, 'substances')
    
    # Define features used for estimation
    features = feat_num + feat_cat_idx + feat_substances_ohe
        
    # Convert all features to Double
    logTime('double')
    feat_select = [ col(c).cast(DoubleType()).alias(c) for c in feat_num ] + feat_cat_idx + feat_substances_ohe
    traindf_temp = traindf.select(col('logprix'), *feat_select)
    testdf_temp = testdf.select(*feat_select)

    # Assemble all features in a single column (name:'features', type:Vector)
    logTime('assemble')
    traindf_temp.show(3)
    vectorAssembler = VectorAssembler(inputCols=features, outputCol='features')
    train_lp_df = vectorAssembler.transform(traindf_temp).select(col('logprix'), col('features'))
    train_lp_df.show(3)
    # Split dataset for train & test
    logTime('randomForest')
    (train, test) = train_lp_df.randomSplit([0.8, 0.2])
    
    # Define estimator 
    rf = RandomForestRegressor(labelCol='logprix', 
                               numTrees=10, 
                               featureSubsetStrategy='auto', 
                               maxDepth=30)
    '''
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.evaluation import RegressionEvaluator
    
    evaluator = RegressionEvaluator(labelCol='logprix', metricName="mae")
    paramGrid = ParamGridBuilder() \
                    .addGrid(rf.maxBins, [16, 32, 64]) \
                    .build()
                                
    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=5)
    
    # Train model 
    model = crossval.fit(train)
    print model.getEstimator().getMaxBins()
    '''
    # Train model  
    logTime('train')
    model = rf.fit(train)
    
    # Evaluate model 
    logTime('transform')
    predictions = model.transform(test)
    
    # Mean Absolute Percentage Error
    def mape_error(log_y_true, log_y_pred): 
        y_true = np.exp(log_y_true)
        y_pred = np.exp(log_y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Compute test error
    logTime('error')
    error = predictions.rdd.map(lambda row: mape_error(row['logprix'], row['prediction'])).mean()
    print error
    

finally:
    spark.stop()
