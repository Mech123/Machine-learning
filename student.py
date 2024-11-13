import argparse
import csv

def load_csvdata(filepath):
    x_feature,y_output = [],[]
    with open(filepath,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            values = list(map(float,row))
            x_feature.append([1.0] + values[:-1])
            y_output.append(values[-1])
    return x_feature,y_output
    
def predicttarget(x_features,weights):
    y_predictions = []
    for x_value in x_features:
        target = sum(w_i*x_i for w_i,x_i in zip(weights,x_value))
        y_predictions.append(target)
        
    return y_predictions
    
    
def estimated_error(y_outputs,predictions):
    errorvalue = sum((y_i-predicted)**2 for y_i,predicted in zip(y_outputs,predictions))
    return errorvalue

def compute_gradientvalue(x_features,y_outputs,predictions):
    gradient = [0.0]*len(x_features[0])
    for i,x_value in enumerate(x_features):
        yi_minus_fxi = y_outputs[i] - predictions[i]
        for j in range(len(x_value)):
            gradient[j] += x_value[j]*yi_minus_fxi
            

    return gradient

def gradient_descent_computation(x_features,y_outputs,etavalue,thresholdvalue):
    weights = [0.0]*len(x_features[0])
    loop_iteration = 0
    previous_errorvalue = float('inf')
    
    while True:
        
        predictions = predicttarget(x_features,weights)
        
        error = estimated_error(y_outputs,predictions)
        
        print(f"{loop_iteration}," + ",".join(f"{w:.9f}" for w in weights) + f",{error:.9f}")
        
        if abs(previous_errorvalue - error) < thresholdvalue:
            break
        
        gradient = compute_gradientvalue(x_features,y_outputs,predictions)
        
        for j in range(len(weights)):
            weights[j] += etavalue * gradient[j]
        
        previous_errorvalue = error
        loop_iteration += 1
        
   
        
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str, required=True, help="Location of the input data file")
    parser.add_argument('--eta',type=float, required=True, help="given Learning rate for the gradient descent")
    parser.add_argument('--threshold', type=float,required=True, help="threshold for the stopping condition mentioned in problem")
    args=parser.parse_args()
    x_features,y_outputs = load_csvdata(args.data)
    gradient_descent_computation(x_features,y_outputs,args.eta,args.threshold)

    
    

