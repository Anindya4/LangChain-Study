from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# modle 1:
model1 = ChatOpenAI(model="gpt-5-nano")


# modle 2:
llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model2 = ChatHuggingFace(llm=llm1)

#Promts

promt1 = PromptTemplate(
    template="Given a text, genarate a concise and detail output in form of study-notes \n {text}",
    input_variables=["text"]
)

promt2 = PromptTemplate(
    template="Generate a small Quiz from the given text \n {text}",
    input_variables=['text']
    
)

promt3 = PromptTemplate(
    template="Merge two given format Text and Quiz into one single document \n notes -> {notes}, quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : promt1 | model1 | parser,
    "quiz" : promt2 | model2 | parser
})

chain2 = promt3 | model1 | parser


final_chain = parallel_chain | chain2

info = """
Linear regression is one of the most widely used and foundational methods in statistics, data analysis, and machine learning. It is designed to model the relationship between a dependent variable (often called the outcome or target) and one or more independent variables (known as predictors or features). The main purpose of linear regression is to understand how the value of the dependent variable changes when any of the independent variables change, assuming all others remain constant. Unlike more complex algorithms, linear regression focuses on identifying a straight-line relationship between variables, making it both simple and interpretable.
In practice, linear regression is applied in many fields such as economics, biology, marketing, and engineering. For example, it can help predict future sales based on advertising budgets, estimate a person's weight from their height, or forecast the impact of temperature on energy consumption. The model learns from existing data by analyzing the patterns and trends that link inputs to outputs. Once trained, it can make predictions for new, unseen data points based on the same learned relationships.
A key strength of linear regression is its interpretability. Each input variable in the model has an associated coefficient, which represents the direction and strength of its influence on the outcome. A positive coefficient means that as the predictor increases, the target variable tends to increase as well, while a negative coefficient suggests the opposite. This feature makes linear regression valuable not only for prediction but also for gaining insights into which factors most strongly affect a result.
Evaluating the performance of a linear regression model involves checking how well its predictions align with actual outcomes. Analysts often use various metrics to assess accuracy, including how close the predicted values are to the observed ones. Additionally, they may examine the residuals the differences between predicted and actual values — to ensure the model captures the main patterns without large systematic errors.
Despite its simplicity, linear regression has some important assumptions. It assumes that the relationship between variables is linear, that errors are random and evenly distributed, and that input variables are not excessively correlated with one another. When these assumptions are reasonably met, linear regression performs very well. However, if the data shows complex, non-linear patterns, other models might be more suitable.
Overall, linear regression remains a cornerstone of data science. It provides a clear and intuitive way to explore relationships between variables, make predictions, and communicate results. Even as more advanced machine learning methods emerge, linear regression continues to be a trusted first step in understanding and modeling real-world data. 
"""

result = final_chain.invoke({'text': info})
print(result)

final_chain.get_graph().print_ascii()



# output:

"""
Study Notes: Linear Regression

- What it is
  - A foundational method to model the relationship between a dependent variable (outcome) and one or more independent variables (predictors).
  - Produces a straight-line relationship, prioritizing simplicity and interpretability.

- Key concepts
  - Dependent variable: the outcome you want to predict.
  - Independent variables: predictors/features that explain or influence the outcome.
  - Coefficients: represent direction and strength of each predictor's influence.
    - Positive coefficient: as the predictor increases, the outcome tends to increase.
    - Negative coefficient: as the predictor increases, the outcome tends to decrease.
  - Intercept (when present in the model): the baseline level of the outcome when predictors are zero (conceptual).

- Types (implicitly in text)
  - Simple linear regression: one predictor.
  - Multiple linear regression: two or more predictors.
  - (Text emphasizes the “one or more” general form.)

- How it works (learning and prediction)
  - The model learns from existing data by analyzing patterns linking inputs to outputs.
  - Once trained, it makes predictions on new, unseen data using the learned relationships.

- Applications (examples)
  - Economics, biology, marketing, engineering.
  - Predict future sales from advertising budgets.
  - Estimate weight from height.
  - Forecast impact of temperature on energy consumption.

- Model evaluation
  - Compare predicted values to actual outcomes to gauge accuracy.
  - Examine residuals (differences between predicted and actual) to detect systematic errors and ensure patterns are captured.

- Assumptions (when the model is reliable)
  - Relationship is linear between predictors and outcome.
  - Errors (residuals) are random and evenly distributed.
  - Predictors are not excessively correlated with one another (avoid strong multicollinearity).

- Strengths
  - Interpretability: clear understanding of how predictors affect the outcome.
  - Simplicity: easy to communicate results and insights.
  - Useful as a baseline or first step in modeling real-world data.

- Limitations
  - Performs best with linear relationships; non-linear patterns may require other models.
  - Sensitivity to violated assumptions and outliers (not detailed in text, but implied by assumptions).

- Practical takeaway / workflow (implicit steps)
  - Specify dependent and independent variables.
  - Fit the model to data to learn coefficients.
  - Check evaluation metrics and residuals to assess fit.
  - Use the model for predictions on new data and interpret predictor effects.

- Quick glossary (from text)
  - Outcome/target vs. predictors/features.
  - Coefficient: direction and strength of a predictor's effect.
  - Residual: difference between predicted and actual value.

Quiz: Linear Regression Quiz

Instructions:  Choose the best answer for each question.

1. What is the main purpose of linear regression?
    a) To find the best-fit curve for a dataset.
    b) To model the relationship between a dependent and one or more independent variables.
    c) To predict the future stock market trends.
    d) To perform complex data analysis techniques.

2. What does the term "coefficient" mean in linear regression?
    a) A random variable with an unknown distribution.
    b) A function that adjusts the dependent variable.
    c) The strength and direction of the relationship between a predictor and a dependent variable.
    d) A measure of the accuracy of a prediction.

3. How does linear regression make an impact on empirical analysis?
    a) By reducing data complexity for specific tasks.
    b) By calculating statistical correlations between variables.
    c) By analyzing and exposing non-linear data patterns.
    d) By helping analysts understand principal variables impacting a dependent variable.

4. Which of the following is NOT an assumption of linear regression?
    a) The relationship between variables is linear.
    b) The errors are random and evenly distributed.
    c) The variables are independent of one another.
    d) The data has a strong, non-linear relationship.

5. What is one of linear regression's key strengths?
    a) Its ability to perform fast and complex calculations.
    b) Its high accuracy and predictive power.
    c) Its comprehensibility and interpretability.
    d) Its flexibility in handling diverse data types.

Answer Key:

1. b
2. c
3. d
4. d
5. c

            +---------------------------+
            | Parallel<notes,quiz>Input |
            +---------------------------+
                 **               **
              ***                   ***
            **                         **
+----------------+                +----------------+
| PromptTemplate |                | PromptTemplate |
+----------------+                +----------------+
          *                               *
          *                               *
          *                               *
  +------------+                 +-----------------+
  | ChatOpenAI |                 | ChatHuggingFace |
  +------------+                 +-----------------+
          *                               *
          *                               *
          *                               *
+-----------------+              +-----------------+
| StrOutputParser |              | StrOutputParser |
+-----------------+              +-----------------+
                 **               **
                   ***         ***
                      **     **
           +----------------------------+
           | Parallel<notes,quiz>Output |
           +----------------------------+
                          *
                          *
                          *
                 +----------------+
                 | PromptTemplate |
                 +----------------+
                          *
                          *
                          *
                   +------------+
                   | ChatOpenAI |
                   +------------+
                          *
                          *
                          *
                +-----------------+
                | StrOutputParser |
                +-----------------+
                          *
                          *
                          *
              +-----------------------+
              | StrOutputParserOutput |
              +-----------------------+

"""