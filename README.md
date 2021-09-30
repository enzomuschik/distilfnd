# DistilFND (Distilled Fake News Detector - Developed by Enzo Muschik)

Multi-modal categorical fake news detector DistilFND trained on Fakeddit benchmark dataset to classify content 
within the 7 types of mis- and disinformation after Wardle (2017). You can read about the defined types of mis- and
disinformation here:

https://firstdraftnews.org/articles/fake-news-complicated/

The entire source code is provided here for research purposes regarding Fake News and its subtypes. Further,
the models and its variants were developed by myself in the context of my master thesis with the title:

*Explanative detection of fake news with deep learning*

In my experiment, I conducted an ablation study to empricially evaluate if certain model components contribute
more or less to high predcticion/classification accuracy scores. In total 4 variants of the model wear developed:

- **Title-DistilFND**: Processing only the Reddit-Post titles to detect subtypes of Fake News
- **Image-DistilFND**: Processing only image data of Reddit-Posts to detect subtypes of Fake News
- **Title-Image-DistilFND**: Processing post title and associated image of Reddit-Post into multi-modale feature representation for detecting subtypes of Fake News
- **DistilFND**: Processing post title, image and first 5 user generated comments to detect subtypes of Fake News

The models were trained on *20%* of the original Fakeddit benchmark dataset and the classification
task is to process the Reddit-Posts, respectively the various modalities according to aforementioned
model variant, and correctly classify input posts into the following classes:

- **Authentic news content**
- **Satire / Parody**
- **Content with false connection**
- **Imposter content**
- **Manipulated content**
- **Misleading content**,

wheres authentic news content and satire/parody is not to be understood as Fake News per se in order
to guard freedom of speech. Further, Fake News is to be defined in the most concise definition after
Alcott and Gentzkow (2017) as

*"[...] news articles that are intentionally and verifiably false, and could mislead readers." (Alcott and Gentzkow (2017), S. 213)

The Fakeddit-Benchmark dataset provides a comprehensive benchmark dataset for training a neural network
model to perform fine-grained categorical Fake News detection. It contains over 700.000 multi-modal sample Reddit-Posts
including image data and over **10 Mio. records** of comment data. All model variants were trained on *20%* of the
complete data set. The training, test and validate splits were generated via the `scikit-learn `train_test_split` function
and stratified in order to maintain the proportional distribution per class of the original dataset. Eeventually, *80%*
of the split data was used for training and *10%* was used for each validation and testing. All models were trained
for 20 epochs if possible, with *early stopping* and a *learning rate scheduler* as well as weightes *cross entropy loss*
to incorporate a highly per class imbalance of the underlying dataset.

# DistilFND Web Appliaction

Additionally, I have developed a **DistilFND web application**, which loads the highest perofming DistilFND model in the backend
and gives users the opportunity to interactively predict a total of 12 sample Reddit-Posts (2 per defined class). Modules, application
python scripts, batch file for easy launching and the requirements.txt file to install needed modules into your python environment
are provided. **All source code is made available for further development and research and is provided within the guidlines
of the MIT license agreement**. Source code for the web application and background information is provided here:

https://github.com/enzomuschik/distilfnd/tree/main/distilfnd_web_app

# Fakeddit Benchmark dataset by *Nakamura et al. (2020)*

The complete Fakeddit benchmark dataset and background information can be found under the following links:

1. https://fakeddit.netlify.app/
2. https://github.com/entitize/Fakeddit
3. https://arxiv.org/abs/1911.03854 (*Nakamura et al. (2020)* research paper)
4. https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view (Download image data)
5. https://drive.google.com/drive/folders/150sL4SNi5zFK8nmllv5prWbn0LyvLzvo (Download comment data)



