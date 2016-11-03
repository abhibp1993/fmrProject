classdef SDC < Car
  
    properties
        senseMask   % A bit-mask to define observable area around the car.
    end
    
    methods
        
        function obj = SDC(id, x, y, h, speed)
            obj = obj@Car(id, x, y, h, speed);
        end
        
        function [nState, act] = transition(obj, state, sensorInput)
            % Implements trivial controller, where speed is given by
            % sensorInput!
            
            act = 'left';
            
            nState.x = state.x;
            nState.y = state.y;
            nState.h = state.h;
            nState.speed = sensorInput;
        end
        
    end
    
end

