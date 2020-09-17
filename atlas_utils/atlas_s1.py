#Import Packages
import numpy as np

class S_1_atlas:
    """
    This class provides an example of the S_1 manifold, and how to traverse it using a sample atlas.
    """
    
    #Initializes an atlas object, sets current position
    #First value in curr_pos_indices must be between 0 and 3, inclusive
    #Second value in curr_pos_indices must be between 0 and 9998, inclusive
    def __init__(self, curr_pos_indices = [0,0]):
        self.atlas = [(np.linspace(-1,1,10000)[1:-1], np.full(9998,1)), (np.full(9998,1), np.linspace(1,-1,10000)[1:-1]), (np.linspace(1,-1,10000), np.full(9998,-1)), (np.full(9998,-1), np.linspace(-1,1, 10000))]
        atlas = self.atlas
        self.curr_pos = [atlas[curr_pos_indices[0]][0][curr_pos_indices[1]], atlas[curr_pos_indices[0]][1][curr_pos_indices[1]]]
        self.curr_pos_indices = curr_pos_indices
    
    #Returns the current coordinate on the atlas
    def get_atlas_coordinates(self):
        return self.curr_pos
    
    #Returns relative indices on map
    def get_atlas_indices(self):
        return self.curr_pos_indices
    
    #Returns the coordinates on the S_1 manifold
    def get_man_coordinates(self):
        if self.curr_pos_indices[0] == 0:
            return ([self.curr_pos[0], np.sqrt(1-(self.curr_pos[0]**2))])
        elif self.curr_pos_indices[0] == 1:
            return ([np.sqrt(1-(self.curr_pos[1]**2)) , self.curr_pos[1]])
        elif self.curr_pos_indices[0] == 2:
            return ([self.curr_pos[0], -np.sqrt(1-(self.curr_pos[0]**2))])
        else:
            return ([np.sqrt(1-(self.curr_pos[1]**2)) , self.curr_pos[1]])

    #Allows a "jump" to occur with a given direction, mag can be positive or negative 
    def jump(self, mag = 1):
        atlas = self.atlas
        if mag >= 0:
            if self.curr_pos_indices[0] != 3:
                #Case where must traverse different maps
                if self.curr_pos_indices[1] + mag > 9997:
                    diff = (mag + self.curr_pos_indices[1]) - 9998
                    self.curr_pos_indices[0] += 1
                    self.curr_pos_indices[1] = diff
                    self.curr_pos = [atlas[self.curr_pos_indices[0]][0][self.curr_pos_indices[1]], atlas[self.curr_pos_indices[0]][1][self.curr_pos_indices[1]]]

                else:
                    self.curr_pos_indices[1] += mag
                    self.curr_pos = [atlas[self.curr_pos_indices[0]][0][self.curr_pos_indices[1]], atlas[self.curr_pos_indices[0]][1][self.curr_pos_indices[1]]]
            else:
                if self.curr_pos_indices[1] + mag > 9997:
                    diff = (mag + self.curr_pos_indices[1]) - 9998
                    self.curr_pos_indices[0] = 0
                    self.curr_pos_indices[1] = diff
                    self.curr_pos = [atlas[self.curr_pos_indices[0]][0][self.curr_pos_indices[1]], atlas[self.curr_pos_indices[0]][1][self.curr_pos_indices[1]]]
                else:
                    self.curr_pos_indices[1] += mag
                    self.curr_pos = [atlas[self.curr_pos_indices[0]][0][self.curr_pos_indices[1]], atlas[self.curr_pos_indices[0]][1][self.curr_pos_indices[1]]]
        else:
            if self.curr_pos_indices[0] != 0:
                #Case where must traverse different maps
                if self.curr_pos_indices[1] + mag < 0:
                    diff = 9998 + (mag + self.curr_pos_indices[1])
                    self.curr_pos_indices[0] -= 1
                    self.curr_pos_indices[1] = diff
                    self.curr_pos = [atlas[self.curr_pos_indices[0]][0][self.curr_pos_indices[1]], atlas[self.curr_pos_indices[0]][1][self.curr_pos_indices[1]]]

                else:
                    self.curr_pos_indices[1] += mag
                    self.curr_pos = [atlas[self.curr_pos_indices[0]][0][self.curr_pos_indices[1]], atlas[self.curr_pos_indices[0]][1][self.curr_pos_indices[1]]]
            else:
                if self.curr_pos_indices[1] + mag < 0:
                    diff = 9998 + (mag + self.curr_pos_indices[1])
                    self.curr_pos_indices[0] = 3
                    self.curr_pos_indices[1] = diff
                    self.curr_pos = [atlas[self.curr_pos_indices[0]][0][self.curr_pos_indices[1]], atlas[self.curr_pos_indices[0]][1][self.curr_pos_indices[1]]]
                else:
                    self.curr_pos_indices[1] += mag
                    self.curr_pos = [atlas[self.curr_pos_indices[0]][0][self.curr_pos_indices[1]], atlas[self.curr_pos_indices[0]][1][self.curr_pos_indices[1]]]
        pass


