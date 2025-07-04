import React, { useState } from 'react';
import { Mic, MicOff } from 'lucide-react';

interface VoiceInputProps {
  isRecording: boolean;
  onRecordingChange: (recording: boolean) => void;
  onTranscription: (transcription: string) => void;
  disabled?: boolean;
}

const VoiceInput: React.FC<VoiceInputProps> = ({ 
  isRecording, 
  onRecordingChange, 
  onTranscription,
  disabled = false
}) => {
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [pendingTranscription, setPendingTranscription] = useState('');

  // Simple sendVoice function like in the HTML file
  const sendVoice = async () => {
    if (disabled || isTranscribing || showConfirmation) {
      return;
    }

    console.log('Voice input triggered');
    setIsTranscribing(true);

    try {
      const response = await fetch('http://localhost:8000/transcribe', {
        method: 'POST'
      });

      console.log('Server response status:', response.status);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Transcription response:', data);
      
      if (data.transcription && data.transcription.trim()) {
        // Show confirmation dialog like in HTML file
        setPendingTranscription(data.transcription.trim());
        setShowConfirmation(true);
      } else {
        console.warn('Empty transcription received');
        alert('❌ Could not transcribe audio.');
      }
    } catch (error) {
      console.error('Voice transcription failed:', error);
      
      let errorMessage = 'Voice input failed: ';
      if (error instanceof TypeError && error.message.includes('fetch')) {
        errorMessage += 'Please check if the server is running on http://localhost:8000';
      } else {
        errorMessage += 'Please try again.';
      }
      
      alert(errorMessage);
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleConfirmTranscription = () => {
    if (pendingTranscription) {
      onTranscription(pendingTranscription);
      setShowConfirmation(false);
      setPendingTranscription('');
    }
  };

  const handleCancelTranscription = () => {
    setShowConfirmation(false);
    setPendingTranscription('');
  };

  const handleVoiceClick = (e: React.MouseEvent) => {
    e.preventDefault();
    if (!disabled && !isTranscribing && !showConfirmation) {
      sendVoice();
    }
  };

  const isButtonDisabled = disabled || isTranscribing || showConfirmation;

  return (
    <div className="flex flex-col items-center gap-2">
      {/* Voice Button */}
      <div className="flex items-center gap-1 sm:gap-2">
        <button
          type="button"
          onClick={handleVoiceClick}
          disabled={isButtonDisabled}
          className="px-3 sm:px-4 py-2 sm:py-3 rounded-lg sm:rounded-xl transition-all duration-200 flex items-center gap-1 sm:gap-2 font-medium hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 min-w-0"
          style={{
            background: isTranscribing
              ? `linear-gradient(135deg, #f59e0b, #fbbf24)`
              : isButtonDisabled
              ? '#9ca3af'
              : `linear-gradient(135deg, var(--accent-tertiary), #4ade80)`,
            color: 'white'
          }}
        >
          {isTranscribing ? (
            <>
              <div className="animate-spin w-3 h-3 sm:w-4 sm:h-4 border-2 border-t-transparent rounded-full border-white"></div>
              <span className="hidden sm:inline text-xs sm:text-sm">Processing...</span>
            </>
          ) : (
            <>
              <Mic size={16} className="sm:w-[18px] sm:h-[18px]" />
              <span className="hidden sm:inline text-xs sm:text-sm">🎤</span>
            </>
          )}
        </button>
        
        {/* Status indicator - Only show on larger screens */}
        {isTranscribing && (
          <div className="hidden sm:flex items-center gap-1" style={{ color: '#f59e0b' }}>
            <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
            <span className="text-xs font-medium">Transcribing...</span>
          </div>
        )}
      </div>

      {/* Confirmation Dialog */}
      {showConfirmation && (
        <div 
          className="mt-2 p-3 rounded-lg border-2 shadow-lg max-w-xs"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            borderColor: 'var(--accent-primary)',
            boxShadow: '0 4px 12px var(--shadow-color)'
          }}
        >
          <p 
            className="text-sm mb-2 font-medium"
            style={{ color: 'var(--text-primary)' }}
          >
            📝 Did you mean:
          </p>
          <p 
            className="text-sm mb-3 italic"
            style={{ color: 'var(--text-secondary)' }}
          >
            "{pendingTranscription}"
          </p>
          <div className="flex gap-2">
            <button
              onClick={handleConfirmTranscription}
              className="flex-1 px-3 py-2 text-xs rounded-lg font-medium transition-all duration-200 hover:scale-105"
              style={{
                background: `linear-gradient(135deg, var(--accent-tertiary), #4ade80)`,
                color: 'white'
              }}
            >
              ✅ Yes
            </button>
            <button
              onClick={handleCancelTranscription}
              className="flex-1 px-3 py-2 text-xs rounded-lg font-medium transition-all duration-200 hover:scale-105"
              style={{
                background: `linear-gradient(135deg, #dc2626, #ef4444)`,
                color: 'white'
              }}
            >
              ❌ No
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceInput;