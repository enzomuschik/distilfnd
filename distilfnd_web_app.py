############ DISTILFND-WEB-APPLICATION (MAIN) ############
"""
Author: Enzo Muschik
Date: 26.09.2021
Master thesis: Explainable detection of fake news with Deep Learning
University: University of Hagen, Hagen, Germany, Faculty for Mathematics and Computer Science
"""

# Setting max number of threads to 8 and importing
# needed modules and classes DataLoader and DistilFND
NUMEXPR_MAX_THREADS = 8
import streamlit as st
import time

############ DISTILFND Model Class ############
# Importing needed modules
from PIL import Image, ImageFile
# Truncating images if too large
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Increasing maximum image pixel size
Image.MAX_IMAGE_PIXELS = 933120000

# Importing Pytorch and transformers from HuggingFace
import torch
from torch import nn
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np

# Importing visualization tools
import seaborn as sns
from pylab import rcParams

# Setting style of seaborn graphic visualization
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
COLOR_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(COLOR_PALETTE))

# Setting parameter figure sizes
rcParams["figure.figsize"] = 12, 8

# Fake News subtypes in order of Fakeddit benchmark dataset labeling
CLASS_NAMES = ["True", "Satire / Parody", "False Conn.", "Impost. Content", "Man. Content", "Mis. Content"]

# Setting web page layout options
st.set_page_config(page_title="DistilFND Web App", page_icon="random", layout="wide", initial_sidebar_state="collapsed")
st.set_option('deprecation.showPyplotGlobalUse', False)

class DistilFND(nn.Module):

    @st.cache()
    def __init__(self, num_classes):
        """
        Constructor function for initializing DistilFND model
        :param num_classes (array with number of classes, here length of 6):
        """
        super(DistilFND, self).__init__()
        # Loading DistilBertModel with pre-trained model weights from English lower case text corpus
        # and assigning to title_module (Title-Feature Extractor)
        self.title_module = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # Loading ResNet34 model with pre-trained model weights from ImageNet 2012 Benchmark dataset
        # and assigning to image_module (Image-Feature Extractor)
        self.image_module = models.resnet34(pretrained="imagenet")
        # Loading DistilBertModel with pre-trained model weights from English lower and upper case text corpus
        # and assigning to comment_module (Comment-Feature Extractor)
        self.comment_module = DistilBertModel.from_pretrained("distilbert-base-cased")
        # Dropout layer to randomly nullify 30% of elements of output tensors --> Useful only in model training
        # Layer is still needed for loading model
        self.drop = nn.Dropout(p=0.3)

        # Fully connected layers (Linear layer) to reshape dimensionality of output tensors ([batch_size, num_classes])
        # Reshaping title feature tensor (768,) --> (1, 6)
        self.fc_title = nn.Linear(in_features=self.title_module.config.hidden_size, out_features=num_classes, bias=True)
        # Reshaping comment feature tensor (768,) --> (1, 6)
        self.fc_comment = nn.Linear(in_features=self.comment_module.config.hidden_size, out_features=num_classes,
                                    bias=True)
        # Reshaping image feature tensor (1, 1000) --> (1, 6)
        self.fc_image = nn.Linear(in_features=1000, out_features=num_classes, bias=True)

        # Final model prediction via Softmax activation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, title_input_ids, title_attention_mask, image, cm_input_ids, cm_attention_mask):
        """
        Forward function feeds input data to layers of DistilFND model --> operational function in
        order to produce a prediction for given Reddit post sample. Forward function accepts input_ids
        and attention_mask from post title and comment data (generated through tokenize function) and
        numeric image vector representation (generated through process_image function)
        :param title_input_ids:
        :param title_attention_mask:
        :param image:
        :param cm_input_ids:
        :param cm_attention_mask:
        :return:
        """
        # Applying title_module onto post_title input_ids and attention_mask
        # Returning title feature tensor of shape (768,)
        title_last_hidden_states = self.title_module(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask,
            return_dict=False
        )
        # List Slicing operation applied to output of last hidden layer of DistilBert model in order to
        # only return tensor representation of aggregated classification output ([CLS] token)
        # and assign to pooled output variable
        title_pooled_output = title_last_hidden_states[0][:, 0, :]
        # Random element nullification of pooled output tensor (not applied during usage of web application)
        # Layer is still needed for loading model
        title_pooled_output = self.drop(title_pooled_output)

        # Output from ResNet34 in shape = (1, 1000) for 1000 classes in correspondence ot ImageNet dataset
        image_output = self.image_module(image)
        # Random element nullification of image output tensor (not applied during usage of web application)
        # Layer is still needed for loading model
        image_output = self.drop(image_output)

        # Applying comment_module onto post_comments input_ids and attention_mask
        # Returning comment feature tensor of shape (768,)
        cm_last_hidden_states = self.comment_module(
            input_ids=cm_input_ids,
            attention_mask=cm_attention_mask,
            return_dict=False
        )
        # List Slicing operation applied to output of last hidden layer of DistilBert model in order to
        # only return tensor representation of aggregated classification output ([CLS] token)
        # and assign to pooled output variable
        cm_pooled_output = cm_last_hidden_states[0][:, 0, :]
        # Random element nullification of pooled output tensor (not applied during usage of web application)
        # Layer is still needed for loading model
        cm_pooled_output = self.drop(cm_pooled_output)

        # Linear layers per title, image and comment tensor output to convert into aligned dimensionality
        # Takes as input the respected title, image and comment tensors and reshapes to shape = (1, 6)
        # for [one sample, 6 defined classes]
        title_condensed = self.fc_title(title_pooled_output)
        image_condensed = self.fc_image(image_output)
        cm_condensed = self.fc_comment(cm_pooled_output)

        # Now, feature vector presentation of different modalities can be merged to one feature representation
        # Merging title and image output tensor to multi-modal feature representation via element-wise maximum method
        fusion = torch.maximum(title_condensed, image_condensed)
        # Adding comment features element-wise to respected multi-modal feature dimensions as 'booster' for
        # most dominant feature representation per class, respectively per subtype of Fake News
        fusion = torch.add(fusion, cm_condensed)

        # Applying Softmax activation function on complete feature vector
        # to return class-specific probability distribution
        return self.softmax(fusion)

    @st.cache(show_spinner=False)
    def load_model(self):
        """
        Loading and initializing best DistilFND model with accuracy of 87,97% trained on 20% of Fakeddit dataset
        :return distilfnd (loaded and trained DistilFND model variable):
        """
        # Initializing DistilFND model class with CLASS_NAMES constant (length of 6 classes)
        distilFND = DistilFND(len(CLASS_NAMES))
        # Loading dictionary state of saved DistilFND and assigning resources to CPU
        distilFND.load_state_dict(torch.load("models/distilfnd.pth", map_location=torch.device("cpu")))

        # Returning loaded and prediction ready DistilFND model
        return distilFND

    def tokenize(self, post_title, post_comments):
        """
        Tokenize function in order to convert raw input data into tokenized feature representations
        :param post_title:
        :param post_comments:
        :return:
        """
        # Loading corresponding DistilBertTokenizer for lower case and lower + upper case
        # English text corpus --> Assigning to corresponding tokeniezr variables
        title_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        comment_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

        # Applying title_tokenizer onto post_title input sequence via encoding_plus function
        # Return is a tokenized sequence of length MAX_LEN = 80
        title_encoding = title_tokenizer.encode_plus(
            post_title,
            # Setting max length to maximum 80 tokens per sequence
            max_length=80,
            # Right-side padding to max length with [PAD] token
            padding="max_length",
            truncation=True,
            # Adding special tokens [CLS], [SEP] and [PAD]
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            # Returning PyTorch tensor
            return_tensors="pt",
        )
        # Try-Except clause for handling exception if comment data non-existent
        try:
            # Applying comment_tokenizer onto comment input sequence via encoding_plus function
            # Return is a tokenized sequence of length MAX_LEN = 80
            comment_encoding = comment_tokenizer.encode_plus(
                post_comments,
                # Setting max length to maximum 80 tokens per sequence
                max_length=80,
                # Right-side padding to max length with [PAD] token
                padding="max_length",
                truncation=True,
                # Adding special tokens [CLS], [SEP] and [PAD]
                add_special_tokens=True,
                return_token_type_ids=False,
                return_attention_mask=True,
                # Returning PyTorch tensors
                return_tensors="pt",
            )
        # Handling ValueError if post has no associated comments
        except ValueError:
            # Initializing post_comments variable with empty string
            post_comments = ""
            # Applying encode_plus function to empty string
            comment_encoding = comment_tokenizer.encode_plus(
                post_comments,
                # Setting max length to maximum 80 tokens per sequence
                max_length=80,
                # Right-side padding to max length with [PAD] token
                padding="max_length",
                truncation=True,
                # Adding special tokens [CLS], [SEP] and [PAD]
                add_special_tokens=True,
                return_token_type_ids=False,
                return_attention_mask=True,
                # Returning PyTorch tensors
                return_tensors="pt",
            )

        # Assigning input_ids and attention_mask tensors from corresponding encode_plus function
        # to matching title and comment encoding variables
        title_input_ids = title_encoding["input_ids"]
        title_attention_mask = comment_encoding["attention_mask"]
        comment_input_ids = comment_encoding["input_ids"]
        comment_attention_mask = comment_encoding["attention_mask"]

        # Returning tokenized encoding input_ids and attention_mask tensors for post title and comments
        return title_input_ids, title_attention_mask, comment_input_ids, comment_attention_mask

    def process_image(self, image):
        """
        Processing function to convert raw input image into feature vector representation
        :param image:
        :return image (processed):
        """
        # Converting raw input image into vector representation via transform function
        transform = transforms.Compose([
            # Resizing raw input image to size of 256
            transforms.Resize(256),
            # Cropping image to size of height x width = 224 x 224
            transforms.CenterCrop(224),
            # Converting image file to PyTorch tensor
            transforms.ToTensor(),
            # Applying normalization to image tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.255]
            )
        ])
        # If image is does not have 3 color channels, convert to RGB image
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Apply transform function on input image --> Returns 3-dimensional image tensor of shape
        # [3, 224, 224] = [color_channel, height in pixel, width in pixel]
        image = transform(image)
        # Apply unsqueeze function to reshape tensor to 4-dimensional tensor carrying the num_images
        # as first position --> [1, 3, 224, 224] = [num_images, color_channels, height, width]
        image = torch.unsqueeze(image, 0)

        # Return processed image which was assigned to original image variable
        return image

############ Dataloader Class ############
# Import needed modules
import pandas as pd
import os
from PIL import Image
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

# Setting style of seaborn graphic visualization
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
COLOR_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(COLOR_PALETTE))

# Setting parameter figure sizes
rcParams["figure.figsize"] = 2, 2

class DataLoader():

    def __init__(self, path, file, image_path):
        """
        DataLoader class is used to fetch and display input data, get next sample
        generate and provide prediction result visualization. Only initialized path
        to source dataframes and path to image folder
        :param path:
        :param file:
        :param image_path:
        """
        # Path to tsv files, e.g. Fakeddit source dataframes
        self.path = path
        # Specification of Fakeddit source tsv file
        self.file = file
        # Path to image folder
        self.image_path = image_path
        # Dataframe attribute occupied after initialize_dataframe function
        self.dataframe = None

    def initialize_dataframe(self):
        """
        Function to load and process tsv file and return a corresponding pandas dataframe
        """
        # Read in Fakeddit source tsv file
        dataframe = pd.read_csv(os.path.join(self.path, self.file), delimiter="\t")

        # Clean dataframe of redundant column
        if "Unnamed: 0" in dataframe.columns:
            dataframe = dataframe.drop(["Unnamed: 0"], 1)

        # Assign processed dataframe to dataframe attribute of DataLoader object
        self.dataframe = dataframe

    def process_comment_display(self, path, file, post_id):
        """
        Process original source comment data tsv file to display input comments in numbered fashion
        :param path:
        :param file:
        :param post_id:
        :return comments:
        """
        # Read in comment data tsv file and assign to pandas dataframe variable
        df_comments = pd.read_csv(os.path.join(path, file), delimiter="\t")

        # Clean comment dataframe of redundant column
        if "Unnamed: 0" in df_comments.columns:
            df_comments = df_comments.drop(["Unnamed: 0"], 1)

        # Find comments associated with specific post with post_id and
        # assign reduced comment dataframe to comment dataframe variable
        df_comments = df_comments.loc[df_comments["submission_id"] == str(post_id)]

        # Processing raw comment data
        # Initialize empty comment list
        comments = []
        # Iterate through first 5 comments per post
        for comment in df_comments["body"][:5]:
            # Drop 'deleted' comments from comment sequence
            if comment == str("[" + "deleted" + "]"):
                continue
            # Drop 'removed' comments from comment sequence
            elif comment == str("[" + "removed" + "]"):
                continue
            # Append first 5 actual comment sequences to comment sequence list
            comments.append(comment)

        # Return comment sequence list
        return comments

    def get_sample(self, post_id):
        """
        Function to fetch post title and comment sequence, post image and ground truth /
        gold label from source Fakeddit benchmark dataset
        :param post_id:
        :return post_title, image, post_comments, gold_label:
        """
        # Fetch and assign dataframe entry with corresponding post_id to sample_post variable
        sample_post = self.dataframe.loc[self.dataframe["id"] == post_id]
        # Fetch and assign post title entry with corresponding index value
        post_title = sample_post["clean_title"][sample_post.index[0]]
        # Try-except clause to fetch corresponding post image form image folder
        try:
            # Trying to open image with post_id and assigning to image variable
            image = Image.open(f"dataset/sample_images/{post_id}.jpg")
        # Handling exception, if image cannot be found
        except FileNotFoundError:
            # Pritning information message
            print("Image could not be retrieved.")

        # Fetching comment entry from sample post dataframe entry with specific index value
        post_comments = sample_post["comments"][sample_post.index[0]]

        # Fetching ground truth label of corresponding post with post_id and assignment to gold_label variable
        gold_label = self.get_gold_label(post_id)

        # Returning post_title, image, post_comments and gold_label variables
        return post_title, image, post_comments, gold_label

    def get_gold_label(self, post_id):
        """
        Fetching corresponding ground truth / gold label for given sample post with
        post Id post_id. Processing numeric class values to readable class names
        :param post_id:
        :return gold_label:
        """
        # Fetching 6-way label of given post via source dataframe and post_id
        sample = self.dataframe.loc[self.dataframe["id"] == post_id]["6_way_label"]
        # Assigning numeric ground truth / gold label value to label variable
        label = sample[sample.index[0]]

        # Converting numeric class value into readable class names
        # If numeric value = 0, assign **True Content** class name to gold_label variable
        if label == 0:
            gold_label = "**True Content**"
        # Else If numeric value = 1, assign **Satire / Parody** class name to gold_label variable
        elif label == 1:
            gold_label = "**Satire / Parody**"
        # Else If numeric value = 2, assign **False Connection** class name to gold_label variable
        elif label == 2:
            gold_label = "**False Connection**"
        # Else If numeric value = 3, assign **Imposter Content** class name to gold_label variable
        elif label == 3:
            gold_label = "**Imposter Content**"
        # Else If numeric value = 4, assign **Manipulated Content** class name to gold_label variable
        elif label == 4:
            gold_label = "**Manipulated Content**"
        # Else If numeric value = 5, assign **Misleading Content** class name to gold_label variable
        elif label == 5:
            gold_label = "**Misleading Content**"
        # If no label available, assign not available to gold_label variable
        else:
            gold_label = "**Not available**"

        # Return fetched and converted gold_label with readable class name per sample post
        return gold_label

    def plot_prob_distribution(self, prediction_probs):
        """
        Function to plot probability distribution bar plot per class for a
        visualized prediction explanation. Enhancing post-hos model interpretability
        by promoting transparency and comprehension through visualization
        :param prediction_probs:
        """
        # Definition readable class names for y-axis of bar plot
        class_names = ["True", "Satire", "False Conn.", "Impost. Content",
                       "Man. Content", "Mis. Content"]

        # Initializing pandas prediction dataframe with dictionary and keys
        # for class names and corresponding probability values per class
        df_pred = pd.DataFrame({
            "class_names": class_names,
            # Iteration through probability values of prediction tensor via list comprehension
            # and assignment of corresponding values per class to probabilities key
            "probabilities": [item for item in prediction_probs.detach().numpy()][0]
        })

        # Initializing horizontal Seaborn bar plot
        # Probability scores / values per class on x-axis
        # Corresponding class names on y-axis
        sns.barplot(x="probabilities", y="class_names", data=df_pred, orient="h")
        # Plotting bar plot via matplotlib module
        # Naming of x- and y-axis
        plt.ylabel("Fake News Subtype")
        plt.xlabel("Probability")
        # Restriting interval for probability values (x-axis)
        plt.xlim([0, 1]);
        # Saving resulting bar plot figure locally to be fetched in main display function
        plot = plt.savefig(f"results/prob_distribution.jpg")

# Load preprocessed dataframe including comments, multimodal_test_public only
# loaded to present input comment data in numbered format
data_loader = DataLoader("dataset", "multimodal_test_comments.tsv", "sample_images")
# Instantiating pandas dataframe
data_loader.initialize_dataframe()


# Fake News subtypes in order of Fakeddit benchmark dataset labeling
CLASS_NAMES = ["True", "Satire", "False Conn.", "Impost. Content", "Man. Content", "Mis. Content"]
# Initializing DistilFND model class
distilfnd = DistilFND(len(CLASS_NAMES))
# Loading best DistilFND model condition from epoch 15 & assigning to CPU ressource
distilfnd = distilfnd.load_model()

# Main application
def main():

    # Setting sidebar title
    st.sidebar.markdown("**Fakeddit - Samples**")

    # Sample drop-down menu for the user to choose specific samples as input
    samples = ["Sample 1 - True Content",
               "Sample 2 - True Content",
               "Sample 3 - Satire / Parody",
               "Sample 4 - Satire / Parody",
               "Sample 5 - False Connection",
               "Sample 6 - False Connection",
               "Sample 7 - Imposter Content",
               "Sample 8 - Imposter Content",
               "Sample 9 - Manipulated Content",
               "Sample 10 - Manipulated Content",
               "Sample 11 - Misleading Content",
               "Sample 12 - Misleading Content"]

    # Instantiating selectbox, user can select and choose specific sample as input
    sample_choice = st.sidebar.selectbox("Choose Sample", samples)

    # Displaying web page header and explanatory abstract
    st.header("**DistilFND - A Multi-modal Categorical Fake News Detector**")
    st.markdown(
        """
        This web application was developed in the context of my master thesis providing an information panel for the classification of Fake News and its subtypes. The underlying neural network DistilFND classifies content
        into a total of 6 classes defined within the 7 subtypes of mis- and disinformation after Wardle (2017). Further, authentic/true content and satire/parody are not to be mistaken for fake news.\n
        For more information on Fake News and its subtypes, refer to the following link: [Wardle (2017): Fake news. It´s complicated](https://medium.com/1st-draft/fake-news-its-complicated-d0f773766c79)
        """)
    st.markdown(
        """
        Source code of DistilFND model and web application is publicly available and can be accessed for research purposes under the following link: [DistilFND Github Repository](https://github.com/H4k4maori/distilfnd)
        """)
    st.markdown(
        """
        **Be aware that the actual model and datasets are loaded in the backend. Hence, give it some time to load! :wink:**
        """)
    st.write("\n")

    # Division of web page layout into three columns, whereas the first and third columns
    # are thrice as large as the second (middle) column
    col1, col2, col3 = st.beta_columns([3, 1, 3])

    # Choice of Sample Input Posts
    # Subtypes are defined within the 7 types of mis- and disinformation after Wardle (2017):
    # https://firstdraftnews.org/articles/fake-news-complicated/
    # Samples are provided within the 6 defined classes of the classification problem
    # Authentic news and Satire / Parody are not to be understood as Fake News, but as authentic content
    # False Connection, Imposter, Manipulating and Misleading Content are regarded as Fake News
    # Sample 1 is default sample and loaded when opening web application
    # All samples are chosen from the publicly available Fakeddit test set, which can be downloaded here:
    # https://fakeddit.netlify.app/
    # All samples are unseen and unknown by DistilFND model
    # Loading specifiied samples and returning post title, associated image, comment data
    # and actual Fakeddit dataset label (gold label) for comparison

    # Loading authentic news content samples
    if sample_choice == "Sample 1 - True Content":
        post_title, image, post_comments, gold_label = data_loader.get_sample("c0gl7r")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "c0gl7r")
    elif sample_choice == "Sample 2 - True Content":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="c0xdqy")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "c0xdqy")
    # Loading Satire / Parody samples
    elif sample_choice == "Sample 3 - Satire / Parody":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="77i4aq")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "77i4aq")
    elif sample_choice == "Sample 4 - Satire / Parody":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="8c9qzv")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "8c9qzv")
    # Loading content with False Connection samples
    elif sample_choice == "Sample 5 - False Connection":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="1q3qqr")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "1q3qqr")
    elif sample_choice == "Sample 6 - False Connection":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="2s0xuj")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "2s0xuj")
    # Loading Imposter Content samples
    elif sample_choice == "Sample 7 - Imposter Content":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="5ll4jw")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "5ll4jw")
    elif sample_choice == "Sample 8 - Imposter Content":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="857qtr")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "857qtr")
    # Loading Manipulated Content samples
    elif sample_choice == "Sample 9 - Manipulated Content":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="chdmd4t")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "chdmd4t")
    elif sample_choice == "Sample 10 - Manipulated Content":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="ctk61yw")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "ctk61yw")
    # Loading Misleading Content samples
    elif sample_choice == "Sample 11 - Misleading Content":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="21wrzf")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "21wrzf")
    elif sample_choice == "Sample 12 - Misleading Content":
        post_title, image, post_comments, gold_label = data_loader.get_sample(post_id="d9erpc")
        comments = data_loader.process_comment_display("dataset", "comments_test_split.tsv", "d9erpc")

    # Tokenization of text sequences and preprocessing image data
    # Returning Input_ids and attention_mask tensors by DistilFND tokenize function
    title_input_ids, title_attention_mask, comment_input_ids, comment_attention_mask = \
        distilfnd.tokenize(post_title, post_comments) # Calling tokenize function with loaded title and comment data
    # Returning processed image data by calling DistilFND process_image function with loaded image
    image_tensor = distilfnd.process_image(image)

    # Column 1: Post input column - Show case input post title, image and first 5 comments of loaded Reddit-Post
    # Displaying additional information and input post title of loaded Reddit sample post
    col1.info("**Input Post**")
    col1.subheader("**Post Title:**")
    col1.markdown("'" + str(post_title) + "'")

    # Displaying additional information and input post image of loaded Reddit sample post
    col1.subheader("**Post Image:**")
    col1.image(image=image, use_column_width=True, clamp=True, channels="RGB")

    # Displaying additional information and first 5 associated input comments of loaded Reddit sample post
    col1.subheader("**User Comments:**")
    # If comment data is non-existent, empty text sequence will be pushed to DistilFND
    # and info message below will be displayed in input comment section
    if len(comments) == 0:
        col1.markdown("**Oops, there are no comments associated with this post!**")
    else:
        # Numbering comment data 1 to max. 5 comments
        counter = 0
        for i in range(len(comments)):
            counter += 1
            col1.markdown(str(counter) + ". " + str(comments[i]))

    # Column 3: Result Column - Displaying results and classification decision
    # Displaying header for result column
    col3.success("**Result**")
    # Displaying actual Fakeddit dataset label per sample (ground truth / gold label)
    # Ground truth label is displayed in Column 3, when input post is loaded
    col3.subheader("**Fakeddit Ground Truth Label:**")
    col3.markdown(f"Sample post was classified as {gold_label} by researchers.")

    # Column 2: Interactive Column - Triggering automatic prediction via integrated DistilFND model
    # Displaying header for interactive column
    col2.info("**Tell Me What It Is!**")
    # If model does not produce a prediction, **Default Value** is assigned
    pred_class = "**Default Value**"
    if col2.button(label="PREDICT POST"):
        # Displaying progress bar to hint running process and better user experience
        progress = col2.progress(0)
        # For-loop to simulate and display advancing progress
        for percent_complete in range(100):
            time.sleep(0.1)
            progress.progress(percent_complete + 1)

        # Putting DistilFND model in evaluation mode. Hence, no parameters are adjusted during prediction process
        # Disabling dropout and regularisation layers
        distilfnd.eval()

        # Starting prediction by providing DistilFND with needed input data
        # Assigning post title input_ids + attention_mask, processed post image
        # and comment (cm) input_ids + attention_mask to distilfnd function
        # Returns prediction probability distribution across 6 defined classes / subtypes of Fake News
        prediction_probs = distilfnd(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask,
            image=image_tensor,
            cm_input_ids=comment_input_ids,
            cm_attention_mask=comment_attention_mask
        )

        # Torch.max function returns highest probability score and additional variable
        # First variable is ignored and highest probability score is assigned to prediction variable
        # Prediction variable represents numeric value between 0 and 5 in correspondence with
        # the 6 defined classes of the classification problem, see https://github.com/entitize/Fakeddit/issues/14
        # 0 = Authentic news content, 1 = Satire / Parody, 2 = False Connection, 3 = Imposter Content,
        # 4 = Manipulated Content and 5 = Misleading Content
        _, prediction = torch.max(prediction_probs, dim=1)

        # Converting numeric prediction value into readable class / subtype name
        # Additional two-way classification in Authentic Content or Fake News, e.g.
        # True news content & Satire / Parody = Authentic Content
        # False Connection, Imposter, Manipulated and Misleading Content = Fake News
        # Access prediction probability with item() function an compare to corresponding numeric class value
        # If equal 0 --> True Content
        if prediction.item() == 0:
            # Assign readable labels to 6-way and 2-way label
            pred_class = "**True Content**"
            two_way_label = "**Authentic Content**"
        elif prediction.item() == 1:
            # Assign readable labels to 6-way and 2-way label
            pred_class = "**Satire / Parody**"
            two_way_label = "**Authentic Content**"
        elif prediction.item() == 2:
            # Assign readable labels to 6-way and 2-way label
            pred_class = "**False Connection**"
            two_way_label = "**Fake News**"
        elif prediction.item() == 3:
            # Assign readable labels to 6-way and 2-way label
            pred_class = "**Imposter Content**"
            two_way_label = "**Fake News**"
        elif prediction.item() == 4:
            # Assign readable labels to 6-way and 2-way label
            pred_class = "**Manipulated Content**"
            two_way_label = "**Fake News**"
        elif prediction.item() == 5:
            # Assign readable labels to 6-way and 2-way label
            pred_class = "**Misleading Content**"
            two_way_label = "**Fake News**"
        # Default: Assign 'not available' default phrase
        else:
            # Assign not available label to 6-way and 2-way label
            pred_class = "**Not available**"
            two_way_label = "**Not available**"

        # Column 3: Result column - After completed prediction, prediction output of DistilFND is displayed
        col3.subheader("**DistilFND Prediction:**")
        # Displaying readable class labels for 6-way and 2-way classification --> Support for efficient
        # user understanding, transparency and interpretability
        col3.markdown(f"Sample post was predicted to be {pred_class} and represents {two_way_label}.")

        # Output of comparison between actual Fakeddit ground truth label and predicted label
        # If predicted label = ground truth label --> prediction is correct
        if pred_class == gold_label:
            result = "**Correct**"
        # If predicted label != ground truth label --> prediction is incorrect
        else:
            result = "**Incorrect**"
        # Displaying result of label comparison
        col3.markdown(f"This prediction was {result}.")

        # Displaying confidence distribution of DistilFND --> Probability distribution for 6 classes with given sample
        col3.subheader("**Confidence Distribution Across Fake News Subtypes:**")
        # Calling plot_prob_distribution function, which saves figure with probability distribution bar plot
        data_loader.plot_prob_distribution(prediction_probs)
        # Loading bar plot images and assigning it to plot variable
        plot = Image.open("results/prob_distribution.jpg")
        # Displaying probability distribution bar plot in Column 3: Result column for enhanced output explainability
        # and interpretability --> Comparison of probability scores between classes, visualized through bar plot
        # Easy knowledge accessibility for end-user and initial step for enhanced model / result explanation
        col3.image(image=plot, use_column_width=True, clamp=True, channels="RGB")

# Hook for main function
if __name__ == "__main__":
    main()


