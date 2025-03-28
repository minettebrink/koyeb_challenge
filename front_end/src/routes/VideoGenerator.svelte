<script lang="ts">
    let prompt: string = '';
    let imageUrl: string = '';
    let imageFile: File | null = null;
    let seed: number | null = null;
    let inferenceSteps: number = 50;
    let guidanceScale: number = 7.5;
    let width: number = 704;
    let height: number = 480;
    let numFrames: number = 161;
    let loading: boolean = false;
    let result: any = null;
    let error: string | null = null;
    let inputMethod: 'url' | 'file' = 'url';

    function validateDimensions() {
        if (width % 32 !== 0 || height % 32 !== 0) {
            throw new Error('Width and height must be divisible by 32');
        }
        if ((numFrames - 1) % 8 !== 0) {
            throw new Error('Number of frames must be divisible by 8 plus 1');
        }
        if (inferenceSteps < 1) {
            throw new Error('Inference steps must be positive');
        }
        if (guidanceScale < 0) {
            throw new Error('Guidance scale must be non-negative');
        }
    }

    async function handleSubmit() {
        loading = true;
        error = null;
        
        try {
            validateDimensions();
            
            const backendUrl = import.meta.env.VITE_BACKEND_URL;
            if (!backendUrl) {
                throw new Error('Backend URL is not configured. Please check your environment variables.');
            }

            console.log('Using backend URL:', backendUrl);

            // Create the request payload
            const payload: any = {
                prompt,
                inference_steps: inferenceSteps,
                guidance_scale: guidanceScale,
                width,
                height,
                num_frames: numFrames
            };

            if (inputMethod === 'url') {
                payload.image_url = imageUrl;
            } else if (imageFile) {
                // Convert file to base64
                const reader = new FileReader();
                const base64Promise = new Promise((resolve, reject) => {
                    reader.onload = () => resolve(reader.result);
                    reader.onerror = reject;
                });
                reader.readAsDataURL(imageFile);
                const base64Data = await base64Promise;
                payload.image_base64 = base64Data;
            }

            // Add optional seed parameter
            if (seed !== null && seed !== undefined) {
                payload.seed = seed;
                console.log('Seed:', seed);
            }

            const url = `${backendUrl}/generate-video`;
            console.log('Making request to:', url);

            const response = await fetch(url, {
                method: 'POST',
                body: JSON.stringify(payload),
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error response:', errorText);
                let errorMessage = 'Failed to generate video';
                try {
                    const errorJson = JSON.parse(errorText);
                    errorMessage = errorJson.detail || errorMessage;
                } catch (e) {
                    errorMessage = errorText;
                }
                throw new Error(errorMessage);
            }

            result = await response.json();
        } catch (e) {
            error = e instanceof Error ? e.message : 'An unknown error occurred';
            console.error('Error details:', e);
        } finally {
            loading = false;
        }
    }

    function handleFileChange(event: Event) {
        const target = event.target as HTMLInputElement;
        if (target.files && target.files[0]) {
            imageFile = target.files[0];
        }
    }
</script>

<div class="container">
    <h1>Video Generator</h1>
    
    <form on:submit|preventDefault={handleSubmit}>
        <div class="form-group">
            <label for="prompt">Prompt:</label>
            <input 
                id="prompt"
                type="text" 
                bind:value={prompt} 
                required
            />
        </div>

        <div class="form-group">
            <fieldset>
                <legend>Image Input Method:</legend>
                <div class="radio-group">
                    <label>
                        <input 
                            type="radio" 
                            bind:group={inputMethod} 
                            value="url"
                        > URL
                    </label>
                    <label>
                        <input 
                            type="radio" 
                            bind:group={inputMethod} 
                            value="file"
                        > File Upload
                    </label>
                </div>
            </fieldset>
        </div>

        {#if inputMethod === 'url'}
            <div class="form-group">
                <label for="imageUrl">Image URL:</label>
                <input 
                    id="imageUrl"
                    type="url" 
                    bind:value={imageUrl} 
                    required
                />
            </div>
        {:else}
            <div class="form-group">
                <label for="imageFile">Upload Image:</label>
                <input 
                    id="imageFile"
                    type="file" 
                    accept="image/jpeg,image/jpg"
                    on:change={handleFileChange}
                    required
                />
            </div>
        {/if}

        <div class="form-group">
            <label for="seed">Seed (optional):</label>
            <input 
                id="seed"
                type="number" 
                bind:value={seed}
            />
        </div>

        <div class="form-group">
            <label for="inferenceSteps">Inference Steps:</label>
            <input 
                id="inferenceSteps"
                type="number" 
                bind:value={inferenceSteps} 
                min="1" 
                max="100"
                required
            />
            <small class="help-text">Controls generation quality (higher = better quality but slower)</small>
        </div>

        <div class="form-group">
            <label for="guidanceScale">Guidance Scale:</label>
            <input 
                id="guidanceScale"
                type="number" 
                bind:value={guidanceScale} 
                step="0.1" 
                min="0" 
                max="20"
                required
            />
            <small class="help-text">Controls prompt adherence (higher = more prompt adherence but potentially less natural)</small>
        </div>

        <div class="form-group">
            <label for="width">Width:</label>
            <input 
                id="width"
                type="number" 
                bind:value={width} 
                min="32" 
                max="1280"
                step="32"
                required
            />
            <small class="help-text">Must be divisible by 32</small>
        </div>

        <div class="form-group">
            <label for="height">Height:</label>
            <input 
                id="height"
                type="number" 
                bind:value={height} 
                min="32" 
                max="720"
                step="32"
                required
            />
            <small class="help-text">Must be divisible by 32</small>
        </div>

        <div class="form-group">
            <label for="numFrames">Number of Frames:</label>
            <input 
                id="numFrames"
                type="number" 
                bind:value={numFrames} 
                min="9" 
                max="257"
                step="8"
                required
            />
            <small class="help-text">Must be divisible by 8 plus 1 (e.g., 161)</small>
        </div>

        <button type="submit" disabled={loading}>
            {loading ? 'Generating...' : 'Generate Video'}
        </button>
    </form>

    {#if error}
        <div class="error">
            {error}
        </div>
    {/if}

    {#if result}
        <div class="result">
            <h2>Generation Result:</h2>
            <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
    {/if}
</div>

<style>
    .container {
        max-width: 800px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    }

    h1 {
        color: #2c3e50;
        margin-bottom: 2rem;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 600;
    }

    .form-group {
        margin-bottom: 1.5rem;
    }

    label {
        display: block;
        margin-bottom: 0.5rem;
        color: #2c3e50;
        font-weight: 500;
    }

    input[type="text"],
    input[type="url"],
    input[type="number"],
    input[type="file"] {
        width: 100%;
        padding: 0.75rem;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        transition: all 0.2s;
        font-size: 1rem;
    }

    input:focus {
        outline: none;
        border-color: #4b5563;
        box-shadow: 0 0 0 3px rgba(75, 85, 99, 0.1);
    }

    fieldset {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0;
    }

    legend {
        color: #2c3e50;
        font-weight: 500;
        padding: 0 0.5rem;
    }

    .radio-group {
        display: flex;
        gap: 2rem;
        margin-top: 0.5rem;
    }

    .radio-group label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        cursor: pointer;
        margin: 0;
    }

    .radio-group input[type="radio"] {
        width: auto;
        margin: 0;
    }

    button {
        background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s;
    }

    button:hover:not(:disabled) {
        background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(31, 41, 55, 0.2);
    }

    button:disabled {
        background: #e2e8f0;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }

    .error {
        color: #dc2626;
        margin-top: 1rem;
        padding: 1rem;
        background-color: #fee2e2;
        border-radius: 8px;
        border: 1px solid #fecaca;
    }

    .result {
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

    .result h2 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }

    pre {
        background-color: #1e293b;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
        font-size: 0.9rem;
    }

    @media (max-width: 640px) {
        .container {
            margin: 1rem;
            padding: 1rem;
        }

        .radio-group {
            flex-direction: column;
            gap: 1rem;
        }
    }

    .help-text {
        display: block;
        margin-top: 0.25rem;
        color: #64748b;
        font-size: 0.875rem;
    }
</style>