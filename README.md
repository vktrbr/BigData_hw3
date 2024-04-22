# Применение ML vs. MLlib в Apache Spark для больших данных
Apache Spark является платформой обработки больших данных, а библиотеки машинного обучения Spark ML и MLlib, предлагают различные подходы и архитектуры для решения задач машинного обучения. Разберем основные различия между ними и их особенности применения.

[Ссылка на презентацию](https://docs.google.com/presentation/d/18FleoMxq_GeQjnaVnV4h6KcEoCKA3QjWmenmFef_LCQ/edit?usp=sharing)

## Архитектура Spark ML и MLlib

### Spark ML
Эта библиотека основана на DataFrame’ах и делает акцент на высокоуровневых абстракциях. 
Spark ML предоставляет более современный API и позволяет легко интегрироваться с другими компонентами Spark, такими как Spark SQL и Spark Streaming.

### MLlib
MLlib — это оригинальная низкоуровневая библиотека машинного обучения, построенная на основе RDD (Resilient Distributed Dataset), ориентированная на производительность и масштабируемость для работы с большими данными.


### Комбинирование
Часто разработчики используют сочетание Spark ML и MLlib, применяя ML для быстрого прототипирования, а MLlib - для продакшн-масштабирования.


## Особенности Spark ML и MLlib 

### Простота Spark ML
- ML предлагает интуитивно понятный API и множество вспомогательных функций для быстрого прототипирования
- DataFrames и Pipelines основные абстракции SparkML

### Гибкость MLlib
- MLlib позволяет более глубоко настраивать модели, производительность.
- В MLlib больше алгоритмов (не все перенесли в Spark ML) 
- Основной компонентой является RDD


### Поддержка
- MLlib: [Main Guide - Spark 3.5.1 Documentation](https://spark.apache.org/docs/latest/ml-guide.html#:~:text=As%20of%20Spark%202.0%2C%20the%20RDD%2Dbased%20APIs%20in%20the%20spark.mllib%20package%20have%20entered%20maintenance%20mode.%20The%20primary%20Machine%20Learning%20API%20for%20Spark%20is%20now%20the%20DataFrame%2Dbased%20API%20in%20the%20spark.ml%20package.)
- Начиная с версии Spark 2.0, пакет spark.mllib перешел в режим поддержки.
- Основным API машинного обучения для Spark теперь является API на основе DataFrame в пакете spark.ml.

## Популярность и востребованность Spark ML и MLlib
Видим, что в последнее время прокси популярность Spark ML в 5 раз выше, чем у MLlib
<img width="1416" alt="Screenshot 2024-04-21 at 21 52 30" src="https://github.com/vktrbr/BigData_hw3/assets/52676181/beed4824-68ad-4bb9-8a78-2c734ba99356">


## Примеры кода для Spark ML
```python
from pyspark.ml.classification import LogisticRegression

# Load training data
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fit the model
mlrModel = mlr.fit(training)

# Print the coefficients and intercepts for logistic regression with multinomial family
print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(mlrModel.interceptVector))
```


## Примеры кода для MLlib
```python
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("data/mllib/sample_svm_data.txt")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# Save and load model
model.save(sc, "target/tmp/pythonLogisticRegressionWithLBFGSModel")
sameModel = LogisticRegressionModel.load(sc,
                                         "target/tmp/pythonLogisticRegressionWithLBFGSModel")
```


