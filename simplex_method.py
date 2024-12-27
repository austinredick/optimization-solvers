import numpy as np

class SimplexMethod():
    def __init__(self, c, A, b, verbose = False):
        self.c = c
        self.A = A
        self.b = b
        self.verbose = verbose
        self.tableau = None

    def construct_tableau(self):
        num_decision_variables = len(self.c)
        num_constraints = len(self.b)
        #create initial tableau based on problem construction with explicit Z column
        slack_vars = np.eye(num_constraints)
        main_rows = np.hstack([self.A, np.eye(num_constraints), np.zeros((num_constraints,1)), self.b.reshape(-1, 1)])
        c_row = np.hstack([self.c, np.zeros(num_constraints), [1], [0]])
        self.tableau = np.vstack((main_rows, c_row))
        if self.verbose:
            print("Initial Tableau:")
            print(self.tableau)

    def pivot(self):
        #find pivot column based on min c_row value
        pivot_column = np.argmin(self.tableau[-1, :-1])
        if self.verbose:
            print("Pivot Column based on value:")
            print(pivot_column, self.tableau[-1,pivot_column])

        if self.tableau[-1, :-1][pivot_column] >= 0:
            print("Optimal solution found.")
            return False

        #find pivot row based on min quotient value of main rows
        quotients = self.tableau[:-1, -1] / self.tableau[:-1,pivot_column]
        pivot_row = np.argmin(quotients)
        '''Considerations:
        what if multiple?
        can pivot value only uses positive values of a and b?
        if so, what if no entries are positive?
        '''
        self.pivot_update(pivot_row, pivot_column)
        if self.verbose:
            print("Updated Tableau:")
            print(self.tableau)
        return True

    def pivot_update(self, p_row, p_col):
        #normalize pivot value to 1 by scaling entire row
        self.tableau[p_row, :] /= self.tableau[p_row, p_col]
        #update other rows so that pivot column values equal 0
        for r in range(self.tableau.shape[0]):
            if r != p_row:
                scalar = self.tableau[r,p_col] * self.tableau[p_row,:]
                self.tableau[r,:] -= scalar
        if self.verbose:
            print("Pivoted Tableau:")
            print(self.tableau)

    def solve(self):
        self.construct_tableau()
        #Call pivot until optimal solution is found
        while self.pivot():
            pass
        #store optimal value of objective
        objective_value = self.tableau[-1, -1]
        #store optimal coeficients of decision variables
        num_decision_variables = len(self.c)
        decision_values = np.zeros(num_decision_variables)
        for i in range(len(self.c)):
            column = self.tableau[:, i]
            if np.count_nonzero(column[:-1]) == 1 and column[-1] == 0:  # Check if it's a unit column
                row_index = np.argmax(column[:-1])
                decision_values[i] = self.tableau[row_index, -1]

        return objective_value, decision_values
