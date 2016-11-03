classdef SDC < Car
  
    properties
        senseMask   % A bit-mask to define observable area around the car.
    end
    
    methods
        function obj = SDC(id, x, y, h, speed)
            obj = obj@Car(id, x, y, h, speed);
            obj.state = {obj.x, obj.y, obj.h};
        end
        
        function [act, output] = transition(obj, state, sensorInput)
        end
    end
    
end

