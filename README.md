# CapstoneTwo
Problem Statement
  The client, HouseX, wants to create a tool for individual home buyers/sellers of residential realestate
  to help predict the value of their property or prospective property. The small startup has a
  team of market researchers that have compiled the provided dataset but are looking for more indepth
  analysis before moving forward. Specifically, the goals set out for us were as follows:
    a. Produce a model with the best predictive accuracy in estimating the value of a property based
      on numerous key features.
    b. Identify which key features positively or negatively impact house price and to what degree.
    c. An added goal would be an additional service for homeowners to evaluate potential
      renovations and estimate the added value they might expect.
      i. I.e., adding on additional square footage to specific rooms or converting a study to a
        bedroom by adding a closet, etc.

Using RMSE as our performance metric, we have determined the Forest Regression model to be the
preferred and best performing model. The linear regression and lasso regression models were very
close, and the ridge model performed the worst of all four. Based on our chosen our model, the four
greatest predictors for a home's value are the home’s age, the general living area square footage, the
exterior quality of the home, the size of the garage.
