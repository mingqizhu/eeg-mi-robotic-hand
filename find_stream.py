from pylsl import resolve_streams

def find_streams():
    print("Scanning for LSL streams (timeout=3s)...")
    streams = resolve_streams(wait_time=3.0)
    
    if len(streams) == 0:
        print("No streams found.")
        print("Make sure your headset software (or mock_stream.py) is running and LSL output is enabled.")
    else:
        print(f"Found {len(streams)} stream(s):")
        for i, stream in enumerate(streams):
            print(f"[{i+1}] Name: {stream.name()}")
            print(f"    Type: {stream.type()}")
            print(f"    Channels: {stream.channel_count()}")
            print(f"    Sample Rate: {stream.nominal_srate()} Hz")
            print(f"    Source ID: {stream.source_id()}")
            print("-" * 30)

if __name__ == "__main__":
    find_streams()
