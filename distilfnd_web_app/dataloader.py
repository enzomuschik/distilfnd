############ Custom DataLoader Class ############
"""
Author: Enzo Muschik
Date: 26.09.2021
Master thesis: Explanatory detection of fake news with Deep Learning
University: University of Hagen, Hagen, Germany, Faculty for Mathematics and Computer Science
"""

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
        plt.savefig(f"results/prob_distribution.jpg")

