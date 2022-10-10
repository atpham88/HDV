#Main HDV model script
#Function inputs and outputs should be pandas dataframes (dfs) where appropriate

def runHDV():
  setKeyParameters()  
  prepareInputs()
  runOptimization()
  analyzeResults()
  pass

#This function sets key parameters we will later vary in sensitivity analyses
def setKeyParameters():
  pass

#Returns dfs for inputs to optimization model
def prepareInputs():
  getTransmissionData() #returns transmission interconnection costs, lengths, etc.
  getSMRData() #returns SMR parameters
  getSolarData() #returns solar parameters
  getStorageData() #returns storage parameters
  getHydrogenData() #returns hydrogen parameters
  
def runOptimization():
  formulateModel()
  solveModel()
  
def analyzeResults():
  pass
  
