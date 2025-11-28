import sys
import optuna

'''
Requires a separate file with an executable function (i.e. the network) for which the parameters can be optimzed
'''


class optuna_optimization():
        
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        
    def load_model(self, class_file_path, config_file_path, class_name='model', dict_names='parameters'):
        
        self.class_name = class_name
        
        sys.path.append(class_file_path)
        sys.path.append(config_file_path)
        #exec("import {}".format(class_name), globals(), globals())
        exec("from {} import *".format(class_name), globals(), globals()) #class of ML model must contain a 'score' method which returns a value (e.g. a loss)
        import configuration # config-file must be called configuration.py
        
        exec("self.params = configuration.{}".format(dict_names))
        exec("self." + self.class_name + "=" + self.class_name + "(self.X, self.y)", globals(), locals())
        
    
    def call_objective(self):
        
        def objective(trial):
            
            type_functions = {'int' : 'trial.suggest_int' , 'float' : 'trial.suggest_float'}

            for key in self.params.keys():
                #print(key + "=" + type_functions[self.params[key][0]] + "('" + key + "'," + str(self.params[key][1]) + "," + str(self.params[key][2]) + ")")
                exec(key + "=" + type_functions[self.params[key][0]] + "('" + key + "'," + str(self.params[key][1]) + "," + str(self.params[key][2]) + ")", globals(), locals())
                

            string_of_vars=""
            for var in self.params.keys():
                if string_of_vars=="":
                    string_of_vars = str(var) + " = " + str(var)
                else:
                    string_of_vars = string_of_vars + "," + str(var) + " = " + str(var)
            
            exec("out_score = self." + self.class_name + ".score(" + string_of_vars + ")", globals(), locals())
            
            return locals()['out_score']
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=100)

        best_params = study.best_params
        
        print(best_params)
        
        return best_params