% This script demonstrates how to use the import functions for the different 
% annotation formats. Furthermore it is shown, how to interpret the
% metadata from the filenames in the SMT-DRUMS database. Please note that
% the different annotation formats in the subfolders annotation_xml and
% annotation_svl are just provided for convenience, the stored onset 
% transcriptions are the same. 

% $Date$ $Revision$ $Author$

% Author: Christian Dittmar (dmr@idmt.fraunhofer.de)
% Created: Jun 2014
% Fraunhofer IDMT. Copyright 2014

%% enable/disable loading
LOAD = 1;

%% get platform specific file separator
fsep = filesep();

%% define input directories
audioDirWAV = ['audio',fsep];
annotDirSVL = ['annotation_svl',fsep];
annotDirXML = ['annotation_xml',fsep];

%% define search suffix
suffixWAV = '*.wav';
suffixSVL = '*.svl';
suffixXML = '*.xml';

%% define the different regexp patterns that will be used to interpret the 
%% metadata from the filename structure
metadataPattern = '[\w_\d]+#(?<instrument>\w+)#(?<type>\w+).\w+|[\w_\d]+#(?<instrument>\w+).\w+';
fileRumpPattern = '(?<testItem>[\w\d_]+)#(?<type>[\w#]+).\w+';
subsetNumPattern = '(?<subSet>\w+)_(?<number>\d+)';
trackTypes = {'MIX','HH','SD','KD'};
numTrackTypes = length(trackTypes);

%% get sub database with perfectly isolated mix samples
fileListWAV = dir([audioDirWAV,suffixWAV]);
fileListSVL = dir([annotDirSVL,suffixSVL]);
fileListXML = dir([annotDirXML,suffixXML]);

%% convert filenames to cells
cellFilenamesWAV = arrayfun(@(x) x.name, fileListWAV,'UniformOutput', false);
cellFilenamesSVL = arrayfun(@(x) x.name, fileListSVL,'UniformOutput', false);
cellFilenamesXML = arrayfun(@(x) x.name, fileListXML,'UniformOutput', false);

if LOAD
  %% go through all audio files and find the corresponding files and annotations
  allItems = [];
  
  for k = 1:length(cellFilenamesXML)
    
    %% get filename
    currentFilename = char(cellFilenamesXML{k});
    
    %% show progress
    disp([num2str(k), ' : ', currentFilename]);
    
    %% and interpret the metadata
    currentMeta = regexpi(currentFilename,fileRumpPattern,'names');
    currentInd = regexpi(currentMeta.testItem,subsetNumPattern,'names');
    
    %% sanity check
    if ~strcmp(currentMeta.type,trackTypes{1})
      continue;
    end
    
    %% initialize metadata of this item
    currentItem = [];
    currentItem.testItem = currentMeta.testItem;
    currentItem.subSet = currentInd.subSet;
    currentItem.number = str2num(currentInd.number);    
    
    %% get all onset annotations and store them
    currentItem.(['onsets',currentMeta.type]) = parseXMLAnnotations([annotDirXML,currentFilename]);
    
    %% find matching annotations via the base name
    indexSVL = find(~cellfun(@isempty,strfind(cellFilenamesSVL,currentMeta.testItem)));
    indexWAV = find(~cellfun(@isempty,strfind(cellFilenamesWAV,currentMeta.testItem)));
    
    %% if there are annotations in SVL format, go through them
    for h = 1:length(indexSVL)
      
      %% get current filename
      currentAnnoname = char(cellFilenamesSVL{indexSVL(h)});
      
      %% parse the annotations
      onsets = parseSVLAnnotations([annotDirSVL,currentAnnoname]);
      
      %% interpret metadata
      annoMeta = regexpi(currentAnnoname,metadataPattern,'names');
      
      %% and store the signals accordingly
      currentItem.(['onsets',annoMeta.instrument,annoMeta.type]) = onsets;
      
    end
    
    for h = 1:length(indexWAV)
      
      %% get current filename
      currentAudioname = char(cellFilenamesWAV{indexWAV(h)});
      
      %% read the audiofile
      [sig,fs] = audioread([audioDirWAV,currentAudioname]);
      
      %% interpret metadata
      audioMeta = regexpi(currentAudioname,metadataPattern,'names');
      
      %% and store the signals accordingly
      currentItem.([audioMeta.instrument,audioMeta.type]) = sig;
      currentItem.fs = fs;
      
    end
    
    %% append to internal set
    allItems{end+1} = currentItem;    

  end
end

%% now, we can select items and look inside
for itemCounter = 1:length(allItems)
  currentItem = allItems{itemCounter};
  
  %% show plots
  for k = 1:numTrackTypes
    subplot(numTrackTypes,1,k);
    if isfield(currentItem,trackTypes{k})
      sig = currentItem.(trackTypes{k});
      sigType = 'ground truth single track';
    else
      sig = currentItem.([trackTypes{k},'train']);
      sigType = 'training onsets';
    end
    hold off
    tAxis = [1:length(sig)]/currentItem.fs;
    plot(tAxis,sig);
    hold on
    onsets = [];
    if isfield(currentItem,['onsets',trackTypes{k}])
      onsets = currentItem.(['onsets',trackTypes{k}]);
      if isstruct(onsets)
        onsets = onsets.onset;
      end
      %% draw each onset
      for h = 1:length(onsets)
        plot([onsets(h) onsets(h)],[-1 1],'r');
      end
    end
    
    title({['subset: ', currentItem.subSet,' number: ', num2str(currentItem.number)],['instrument: ',trackTypes{k},' type: ',sigType]})
    axis tight;
    drawnow;
  end
  1;
end

