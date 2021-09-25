############ DISTILFND Class ############
"""
Author: Enzo Muschik
Date: 26.09.2021
Master thesis: Explainable detection of fake news with Deep Learning
University: University of Hagen, Hagen, Germany, Faculty for Mathematics and Computer Science
"""

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

class DistilFND(nn.Module):

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

    def load_model(self):
        """
        Loading and initializing best DistilFND model with accuracy of 87,97% trained on 20% of Fakeddit dataset
        :return distilfnd (loaded and trained DistilFND model variable):
        """
        # Initializing DistilFND model class with CLASS_NAMES constant (length of 6 classes)
        distilFND = DistilFND(len(CLASS_NAMES))
        # Loading dictionary state of saved DistilFND and assigning resources to CPU
        distilFND.load_state_dict(torch.load("dataset/models/distilfnd_model.pth", map_location=torch.device("cpu")))

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

