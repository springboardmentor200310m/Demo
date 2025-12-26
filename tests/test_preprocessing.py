import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import load_audio, generate_mel_spectrogram

class TestPreprocessing(unittest.TestCase):

    @patch('librosa.load')
    def test_load_audio_success(self, mock_load):
        # Setup mock
        mock_audio = np.zeros(22050)
        mock_sr = 22050
        mock_load.return_value = (mock_audio, mock_sr)
        
        # Test
        y, sr = load_audio('dummy.wav')
        
        # Assert
        self.assertIsNotNone(y)
        self.assertEqual(sr, 22050)
        mock_load.assert_called_once()

    @patch('librosa.load')
    def test_load_audio_failure(self, mock_load):
        # Setup mock to raise exception
        mock_load.side_effect = Exception("File not found")
        
        # Test
        y, sr = load_audio('nonexistent.wav')
        
        # Assert
        self.assertIsNone(y)
        self.assertIsNone(sr)

    @patch('librosa.feature.melspectrogram')
    @patch('librosa.power_to_db')
    def test_generate_mel_spectrogram(self, mock_power_to_db, mock_melspectrogram):
        # Setup mock
        mock_y = np.zeros(22050)
        mock_sr = 22050
        mock_S = np.zeros((128, 100))
        mock_S_dB = np.zeros((128, 100))
        
        mock_melspectrogram.return_value = mock_S
        mock_power_to_db.return_value = mock_S_dB
        
        # Test
        S_dB = generate_mel_spectrogram(mock_y, mock_sr)
        
        # Assert
        self.assertIsNotNone(S_dB)
        self.assertEqual(S_dB.shape, (128, 100))
        mock_melspectrogram.assert_called_once()
        mock_power_to_db.assert_called_once()

if __name__ == '__main__':
    unittest.main()
