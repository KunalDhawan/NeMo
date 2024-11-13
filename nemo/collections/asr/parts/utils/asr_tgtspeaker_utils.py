import numpy as np
from lhotse import SupervisionSet, SupervisionSegment, MonoCut, Recording, CutSet


def get_separator_audio(freq, sr, duration, ratio):
    # Generate time values
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Generate sine wave
    y = np.sin(2 * np.pi * freq * t) * 0.1

    y[:int(sr * duration * ratio )] = 0
    y[-int(sr * duration * ratio ):] = 0
    return y

def get_query_cut(cut):
    '''
    Extract query from the cut and saved as a separate cut

    Args:
        cut: An audio cut. The cut should contain keys "query_audio_filepath", "query_offet", "query_duration"

    Returns:
        query_cut: a cut containing query information
    '''    
    if 'query_audio_filepath' in cut.custom:
        query_rec = Recording.from_file(cut.query_audio_filepath)
        query_sups = [SupervisionSegment(id=query_rec.id+'_query'+str(cut.query_offset)+'-'+str(cut.query_offset + cut.query_duration), recording_id = query_rec.id, start = 0, duration = cut.query_duration, speaker = cut.query_speaker_id)]
        query_cut = MonoCut(id = query_rec.id +'_query'+str(cut.query_offset)+'-'+str(cut.query_offset + cut.query_duration),
                            start = cut.query_offset,
                            duration = cut.query_duration,
                            channel = 0,
                            recording = query_rec,
                            supervisions = query_sups)
        return query_cut
    else:
        query_rec = cut.recording
        query_sups = [SupervisionSegment(id=cut.id+'_query_dummy', recording_id = query_rec.id, start = 0, duration = 0, speaker = None)]
        query_cut = MonoCut(id = cut.id +'_query_no_ts_'+str(cut.start)+'_'+str(cut.duration),
                            start = 0,
                            duration = 0,
                            channel = 0,
                            recording = query_rec,
                            supervisions = query_sups)
        return query_cut
    
