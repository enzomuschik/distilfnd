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
from PIL import Image
# DataLoader to load & prepare input and generate prediction output
from dataloader import DataLoader
# DistilFND loads model itself and prepares input data
from distilfnd import DistilFND
import torch
import time

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
distilfnd.load_state_dict(torch.load("models/distilfnd.pth", map_location=torch.device("cpu")))

# Main application
def main():
    # Setting web page layout options
    st.set_page_config(page_title="DistilFND Web App", page_icon="random", layout="wide", initial_sidebar_state="collapsed")
    st.set_option('deprecation.showPyplotGlobalUse', False)

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


