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
            
            act = 'straight';
            
            nState.x = obj.state.x;
            nState.y = obj.state.y;
            nState.h = obj.state.h;
            nState.speed = obj.state.speed;
        end
        
    end
    
end

