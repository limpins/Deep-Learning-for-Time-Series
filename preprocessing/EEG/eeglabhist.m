% 10-Apr-2019
% EEGLAB 数据预处理流程
% ------------------------------------------------

tic;
dipfit = 'D:\\Softwares\\eeglab7\\plugins\\dipfit2.3\\standard_BESA\\standard-10-5-cap385.elp';

for id=1:69
    eeglab;
    EEG = pop_importdata('dataformat', 'matlab', 'nbchan', 3, 'data', ['../../Data/depression/', int2str(id), '.mat'], 'setname','depression', 'srate',1000, 'pnts', 0, 'xmin', 0);
    EEG.setname='depression';
    EEG = eeg_checkset( EEG );
    EEG=pop_chanedit(EEG, 'lookup', dipfit, 'load', {'../../Data/depression/loc_data.ced' 'filetype' 'autodetect'});
    EEG = eeg_checkset( EEG );
    % EEG = pop_resample( EEG, 512);
    EEG = pop_eegfiltnew(EEG, 0.5, 100, 3380, 0, [], 0);
    EEG = eeg_checkset( EEG );
    EEG = pop_eegfiltnew(EEG, 49, 51, 1690, 1, [], 0);
    EEG = eeg_checkset( EEG );
    EEG = pop_rmbase(EEG, [], []);
    EEG = eeg_checkset( EEG );
    % EEG = pop_runica(EEG, 'extended', 1, 'interupt', 'on');
    % EEG = eeg_checkset( EEG );
    % EEG = pop_subcomp( EEG, [], 0);
    EEG = pop_rejcont(EEG, 'elecrange',[1:3] ,'freqlimit',[20 40] ,'threshold',10,'epochlength',0.5,'contiguous',4,'addlength',0.25,'taper','hamming');
    EEG = eeg_checkset( EEG );
    % EEG = pop_prepPipeline(EEG, struct('ignoreBoundaryEvents', true));
    % EEG = eeg_checkset( EEG );
    % EEG = pop_prepPipeline(EEG, struct('detrendChannels', [1  2  3], 'detrendCutoff', 1, 'detrendStepSize', 0.02, 'detrendType', 'High Pass'));
    % EEG = eeg_checkset( EEG );
    % EEG = pop_prepPipeline(EEG, struct('lineNoiseChannels', [1  2  3], 'lineFrequencies', [60  120  180  240  300  360  420  480], 'Fs', 1000, 'p', 0.01, 'fScanBandWidth', 2, 'taperBandWidth', 2, 'taperWindowSize', 4, 'pad', 0, 'taperWindowStep', 1, 'fPassBand', [0  500], 'tau', 100, 'maximumIterations', 10));
    % EEG = eeg_checkset( EEG );
    % EEG = pop_prepPipeline(EEG, struct('cleanupReference', false, 'keepFiltered', false, 'removeInterpolatedChannels', true));
    % EEG = eeg_checkset( EEG );

    % data = mapminmax(EEG.data, -1, 1).';
    data = EEG.data.';
    save(['E:\\Github\\Deep-Learning-for-Time-Series\\Data\\new_eeg\\', int2str(id), '.mat'], 'data');
end

toc;
