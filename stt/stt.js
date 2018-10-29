window.SpeechRecognition =
  window.webkitSpeechRecognition || window.SpeechRecognition;

if ('SpeechRecognition' in window) {
  const recognition = new window.SpeechRecognition();
  recognition.lang = 'ko-KR';
  recognition.continuous = true;
  recognition.interimResults = false;
  
  recognition.onresult = event => {
    const last = event.results.length - 1;
    const finalTranscript = event.results[last][0];
    const { transcript } = finalTranscript;
    // transcript 가 최종으로 기록된 음성을 텍스트로 변환한 것.
  };
}
