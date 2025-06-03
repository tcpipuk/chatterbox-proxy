# 🎙️ Chatterbox TTS Proxy

Ever wanted to use ChatGPT's voice features with your own custom voices? This proxy makes it happen!
It's a simple bridge that lets any app using OpenAI's TTS API tap into Chatterbox's awesome voice
cloning powers. No code changes needed - just point your app at this proxy and you're good to go! 🚀

## ✨ Why Use This Proxy?

* Drop-in Replacement - Works with any OpenAI TTS client - just swap the endpoint URL
* Zero-shot Voice Cloning - Clone any voice from a single audio sample
* Emotional Control - Make your AI sound happy, sad, or anywhere in between
* Production Ready - High-quality audio that's ready for real-world use
* Easy Deployment - One command to get everything running

## 🎯 Chatterbox TTS

Chatterbox is Resemble AI's open-source TTS engine that's seriously impressive:

* Clone any voice from a single audio sample (yes, really)
* Control how emotional the voice sounds
* Generate studio-quality audio
* Works with WAV, MP3, and FLAC files

This project uses [bhimrazy's Chatterbox TTS server](https://github.com/bhimrazy/litserve-examples/tree/main/chatterbox-tts)
from [Docker Hub](https://hub.docker.com/r/bhimrazy/chatterbox-tts). It's the perfect starting point
for voice cloning experiments!

## 🔄 Adapter Proxy

The adapter translates between OpenAI's TTS API format and Chatterbox's more powerful capabilities:

* Converts OpenAI requests to Chatterbox format
* Manages voice presets and settings
* Handles errors gracefully
* Makes everything just work™

Here's how it works:

**Your app sends this** (standard OpenAI format):

```json5
{
  "model": "tts-1", #this value is ignored
  "voice": "en-uk-heart",
  "input": "Hello, this is a test."
}
```

**The proxy converts it to this** (Chatterbox format):

```json5
{
  "text": "Hello, this is a test.",
  "audio_prompt": "reference/voice_sample.wav",
  "cfg": 0.4,          # Controls generation quality
  "exaggeration": 0.6,  # Controls emotional intensity
  "temperature": None, # Only needed when an audio prompt isn't provided
}
```

The magic happens with these settings:

* **`audio_prompt`** - Your reference audio file (WAV, FLAC, or MP3)
* **`exaggeration`** - How emotional the voice should be (0.0-1.0)
* **`cfg`** - How closely to match the reference voice (0.0-1.0)
* **`temperature`** - If you have no reference file, this controls the randomness of the voice

## 🎭 Voice Cloning

Want to add your own voice? It's super easy! Here's how:

1. Drop your audio file in the `reference/` folder
   (5-45 seconds will work, but I recommend at least 15-20 seconds for accuracy)
2. Add a new preset to `adapter.py`
3. Tweak the settings:
   * `exaggeration`: Crank it up (0.8-1.0) for more dramatic speech
   * `cfg`: Higher values (0.7-0.9) for better voice matching

That's it! The proxy handles all the complicated stuff behind the scenes. 🎯

## 🚀 Setup & Run

### Docker Compose

Fire it up with one command:

```bash
docker-compose up -d
```

### 📁 Folder Structure

```bash
.
├── models/       # Hugging Face models (cached)
├── reference/    # Your voice samples go here
├── voices.yml    # Voice preset configurations
└── docker-compose.yml
```

### 🎭 Voice Configuration

The `voices.yml` file lets you configure your voice presets. Each preset can use either:

* An audio sample for voice cloning (recommended)
* Temperature-based generation (when no audio sample is available)

Example configuration:

```yaml
# Voice with audio sample
stewie:
  name: Stewie
  audio_prompt: stewie.mp3
  exaggeration: 0.8  # How emotional (0.0-1.0)
  cfg: 0.4          # Voice matching quality (0.0-1.0)

# Voice without audio sample
random-voice:
  name: Random Voice
  temperature: 0.7  # Voice randomness (0.0-1.0)
  exaggeration: 0.5
  cfg: 0.5
```

To use your own voice configuration:

1. Create a `voices.yml` file in your project directory
2. Mount it in the container using the volume in `docker-compose.yml`
3. Optionally override the path using the `VOICES_FILE` environment variable

The proxy will automatically reload the voice presets when the container starts.

### 🧪 Quick Test

You can test the proxy directly using curl:

```bash
curl -X POST http://your-server:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "voice": "stewie",
    "input": "How does it feel to be the least cultured person at a bus station?"
  }' \
  --output test.wav
```

This will:

1. Send a request to the proxy
2. Use the "stewie" voice preset
3. Save the generated audio as `test.wav`

You can then play the file to hear the result. Try different voices and text to experiment!

## ⚖️ Ethical Considerations

We strongly encourage responsible use of this technology:

* Use your own voice or public domain voices for personal projects
* Get explicit permission before cloning someone else's voice
* Be transparent when using AI-generated voices
* Consider the impact on voice actors and content creators

While we can't control how others use this tool, we don't condone:

* Impersonating others without consent
* Creating misleading or harmful content
* Using cloned voices for fraud or deception
* Any other malicious or unethical purposes

Remember: With great power comes great responsibility! 🕷️

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
