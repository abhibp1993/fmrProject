% Author: Abhishek Kulkarni
% Last Modified: 3 Nov 2016

classdef (Abstract) StateMachine
   
    properties
       state
    end
    
    methods (Abstract)
        step(obj, sensorInput)
        transition(obj, state, sensorInput)
    end
    
end

