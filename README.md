# Foxintelligence Data Analysis Project

![Foxintelligence Logo](assets/Foxintelligence.png)

## Introduction

This project is designed to analyze and extract insights from digital market data with a focus on the impact of Covid confinement in France. The analysis aims to provide app editors with valuable insights that can enhance their understanding of market dynamics during the confinement period.

## Project Structure

The project is structured into several Python scripts and Jupyter notebooks that perform various tasks such as data processing, image matching, and text matching:

- `data_processor.py`: Handles data cleaning and preparation.
- `text_matcher.py`: Matches product names to titles using fuzzy string matching.
- `image_matcher.py`: Compares images for similarity using perceptual hashing.
- `image_hasher.py`: Utility script for generating image hashes.
- `main.py`: Orchestrates the running of the data processing, text matching, and image matching scripts.

Additionally, Jupyter notebooks are provided for exploratory data analysis and visualization:

- `data_exploration.ipynb`: Explores the data to identify trends and patterns.
- `viz.ipynb`: Visualizes the results in a meaningful way to draw actionable insights.

## Setup and Installation

To set up the project environment, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourrepository/foxintelligence.git
