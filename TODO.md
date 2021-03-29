get_loss: was ist mit mode='test' etc für dropout? kann ich eifnach ypred=self[-1].a nehmen?

regularizers conv net

model.loss(xtrain, ytrain) lässt das memory volllaufen -> nach und nach daten durch das netz schieben und loss accumulieren ist die einzige möglichkeit
