# Copyright (c) OpenMMLab. All rights reserved.
from .audio_recognizer import AudioRecognizer
from .base import BaseRecognizer
from .recognizer2d import Recognizer2D, Recognizer2DForward
from .recognizer3d import Recognizer3D
from .recoginizer_two_stream import RecoginizerTwoStream
from .recoginizer_two_stream_fuse_noguide import RecoginizerTwoStream_fuse_noguide
from .recoginizer_two_stream_nofuse import RecoginizerTwoStream_nofuse
from .recognizer_channel_wise import Recognizer_cta
from .recoginizer_two_stream_hmj import RecoginizerTwoStreamVit
from .recoginizer_two_stream_hmj_jc import RecoginizerTwoStreamVitJC
from .recoginizer_two_stream_simplecat import RecoginizerTwoStream_simplecat
from .recoginizer_two_stream_daft import RecoginizerTwoStream_DAFT
from .recoginizer_two_stream_daft_qual import RecoginizerTwoStream_DAFT_QUAL
from .recoginizer_two_stream_daft_qual_2fc import RecoginizerTwoStream_DAFT_QUAL_2fc
from .recoginizer_two_stream_daft_qual_5infer import RecoginizerTwoStream_DAFT_QUAL_5infer
# from .recoginizer_two_stream_seg_hmj import RecoginizerTwoStreamSeg
__all__ = ['BaseRecognizer', 'Recognizer2D', 'Recognizer3D', 'AudioRecognizer',
           'RecoginizerTwoStream','RecoginizerTwoStream_nofuse',
           'RecoginizerTwoStream_fuse_noguide', 
           'RecoginizerTwoStream_simplecat','RecoginizerTwoStream_DAFT',
           'Recognizer_cta', 'Recognizer2DForward',
           'RecoginizerTwoStreamVit','RecoginizerTwoStreamVitJC',
           'RecoginizerTwoStream_DAFT_QUAL','RecoginizerTwoStream_DAFT_QUAL_2fc',
           'RecoginizerTwoStream_DAFT_QUAL_5infer',
           ]
