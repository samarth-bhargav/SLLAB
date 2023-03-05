## SLLAB
## Inspiration
We are interested in the effects of urbanization on the environment around us, particularly light pollution. Artificial light has significant affects on a humans and the ecosystem around us so it is imperative we find a efficient and accurate to detect light pollution around the world.  

## What it does
We created a highly accurate machine learning model that predicts light pollution levels. Additionally, we also used an linear regression model and data analysis techniques to accurately forecast the amount of light pollution that will be present in a given area. We also created an interactive website using HTML  and CSS where one can input latitude, longitude, and year to learn the amount of light pollution in your area. 

## How we built it
To create the machine learning model, we utilized Pytorch's recurrent neural network. Using data scraped from images taken by the NOAA 2020 and Suomi NPP satellites. We obtained monthly data from the past 10 years and trained the data using the ReLU activation function and two fully connected linear layers. We took advantage of the time series data by splicing it into segments of 10 and performed forward and backward propagation on each segment.

## Challenges we ran into
Challenges we faced were related to the construction of the neural network. The nature of the neural network required the use of a fast learning rate, but the fast learning rate lead to convergence issues, where each of the sequences would map to the same output. We fixed this by segmenting the data to create more training data from the data we inputted. We also faced issues on the front end of the application, using Flask to import both our linear regression. 

## Accomplishments that we're proud of
We successfully created a machine learning and linear regression model that is accurate and an interactive website for users. Furthermore, this can easily be implement for real world applications. 

## What we learned
We learned how to use Flask, HMTL, and CSS. In addition, we also learned how to use Pytorch to create recurrent neural networks, and also how to scrape data from a website. We also learned how to optimally tweak the hyperparameters of our machine learning model to have convergence occur at just the right pace. 

## What's next for SLLAB
We hope to expand this model by continually updating with recent data and collecting data all across the US and internationally. 