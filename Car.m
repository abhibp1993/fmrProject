% Author: Abhishek Kulkarni
% Last Modified: 1 Nov 2016

classdef Car < StateMachine
    
    properties
        id      % unique id of the car
        x       % x-position (row)
        y       % y-posiiton (col)
        h       % heading (Orientation object)
        speed   % speed in steps/unit-time
        
    end
    
    
    methods
        
        function obj = Car(id, x, y, h, speed)
            obj.id = id;
            obj.x = x;
            obj.y = y;
            obj.h = h;
            obj.speed = speed;
            
            obj.senseMask = ones(5, 5);
        end
        
        function pos = position(obj)
            pos = [obj.x, obj.y];
        end
        
        function pos = pose(obj)
            pos = [obj.x, obj.y, obj.h];
        end
        
        function output = step(obj, sensorInput)
            
        end
        
        function [] = print(obj)
            fprintf('Car id = %d, X = %d, Y = %d\n', obj.id, obj.x, obj.y);
            % TODO: find error in printing. seems to be printing array of cars 'sideways'
        end
       
    end
    
end
