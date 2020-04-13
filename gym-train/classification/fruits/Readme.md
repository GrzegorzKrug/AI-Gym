## Fruits classification
Data set contains around
* 40k training images
* 20k test images

Learning stopped after 2 consecutive epochs with accuracy at least **98%** on training set.

Due to normalization images take more memory. Float uses more bits then integer.

#
### First run

* Using 15% of training samples
* Using 15% of test samples

First 25 fruits classified successfully.
![Validation](./last-validation.png)

![Graph](./Classification-of-120-fruits.png)

#
### Image conversion to float16
Using all images.

![Validation](./fruits-120-all_fruits_float16-with-98stop-1586819948-validation_example.png)

![Graph](./fruits-120-all_fruits_float16-with-98stop-1586819948-graph.png)

#
### Image conversion to float32
Using 70% of dataset

![Validation](./fruits-120-70%25_fruits_float32-with-98stop-1586820339-validation_example.png)

![Graph](./fruits-120-70%25_fruits_float32-with-98stop-1586820339-graph.png)

#
### Image conversion to float64
Using only 25% of dataset.

![Validation](./fruits-120-25%25_fruits_float64-with-98stop-1586820895-validation_example.png)

![Graph](./fruits-120-25%25_fruits_float64-with-98stop-1586820895-graph.png)

#
#### Dataset used in learning
[Kaggle-Fruits](https://www.kaggle.com/moltean/fruits)
