import ReactAudioPlayer from 'react-audio-player';
import myAudio from '../../music/audio0.wav'
export default function Audio(){

  const musicTracks = [
    {
      name: "audio",
      src: "../../music/audio.wav"
    }
  ]

  return(
    <div>
        <ReactAudioPlayer
          style={{ width: "500px" , borderRadius: "0.5rem" }}
          controls
          layout="horizontal"
          src={myAudio}
          onPlay={(e) => console.log("onPlay")}
          showSkipControls={false}
          showJumpControls={false}
          header={`Now playing: ${musicTracks.name}`}
        />
    </div>
  )
}
