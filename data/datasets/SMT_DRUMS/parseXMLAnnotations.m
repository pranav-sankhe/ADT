function [ annotations ] = parseXMLAnnotations( xmlFile )

% [ annotations ] = parseXMLAnnotations( xmlFile )
% Function can read the read annotation xml files from Fraunhofer IDMT
%
% Input:
%   xmlFile: The complete path to the svl-file
% 
% Output:
%   annotations.instrName: instrument short name
%   annotations.instrCode: instrument short code
%   annotations.pitch: the MIDI pitch
%   annotations.onset: onset in seconds
%   annotations.offset: offset in seconds
%

% $Date$ $Revision$ $Author$

% Author: Christian Dittmar (dmr@idmt.fraunhofer.de)
% Created: Jun 2014
% Fraunhofer IDMT. Copyright 2014

dom = xmlread(xmlFile);

events = dom.getElementsByTagName('event');
numEvents = events.getLength;

instrName = [];
pitch = [];
onset = [];
offset = [];

for p = 1:numEvents
  
  instrName{p} = char(events.item(p-1).getElementsByTagName('instrument').item(0).getTextContent);
  pitch(p) = str2num(events.item(p-1).getElementsByTagName('pitch').item(0).getTextContent);
  onset(p) = str2num(events.item(p-1).getElementsByTagName('onsetSec').item(0).getTextContent);
  offset(p) = str2num(events.item(p-1).getElementsByTagName('offsetSec').item(0).getTextContent);
  
end

%% prepare output struct
annotations.instrName = [];
annotations.pitch = [];
annotations.onset = [];
annotations.offset = [];

%% stuff everything into output container
annotations.instrName = instrName;
annotations.pitch = pitch;
annotations.onset = onset;
annotations.offset = offset;

end

