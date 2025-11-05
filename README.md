# D501 PA Task 2: MLOps pipeline deployment with FastAPI

This is the final project for the WGU course D501, submitted on Udacity. The edits are my own and are intended only for my own submission for grading. Any unauthorized use of my edits is explicitly forbidden and violates WGU's Academic Integrity policy.

## Links

- GitHub Repository - [de-taylor/D501-MLOps-Project2](https://github.com/de-taylor/D501-MLOps-Project2)
- Heroku App - [D501-Heroku-Test](https://d501-heroku-test-35f2d874348f.herokuapp.com/)

## Model Card

To learn more about this LogisticRegression model, please review the [Model Card](https://github.com/de-taylor/D501-MLOps-Project2/blob/main/model_card_LogReg.md)

## Project Instructions

### Environment Set up (pip or conda)
- [ ] Option 1: use the supplied file `environment.yml` to create a new environment with conda
- [x] Option 2: use the supplied file `requirements.txt` to create a new environment with pip
    
#### Repositories
- [x] Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
- [x] Connect your local git repo to GitHub.
- [x] Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    - [x] Make sure you set up the GitHub Action to have the same version of Python as you used in development.

### Data
- [x] Download census.csv and commit it to dvc.
- [x] This data is messy, try to open it in pandas and see what you get.
- [x] To clean it, use your favorite text editor to remove all spaces.

### Model
- [x] Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
- [x] Write unit tests for at least 3 functions in the model code.
- [x] Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
- [x] Write a model card using the provided template.

### API Creation
- [x]  Create a RESTful API using FastAPI this must implement:
    - [x] GET on the root giving a welcome message.
    - [x] POST that does model inference.
- [x] Deployed to Heroku
