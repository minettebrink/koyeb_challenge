<script>
  import { onMount } from "svelte";

  let apiData = [];

  // Fetch data when the component is mounted
  onMount(async () => {
      try {
          const res = await fetch("http://127.0.0.1:8000/items/58?q=somequery");
          if (res.ok) {
              const json = await res.json();
              apiData = json; // Update apiData directly
          } else {
              console.error("Failed to fetch data from FastAPI.");
          }
      } catch (error) {
          console.error("Error fetching data:", error);
      }
  });
</script>

<main class="p-4">
  <h1 class="text-2xl font-bold mb-4">Data from FastAPI</h1>
  <ul>
      {apiData.item_id}
  </ul>
</main>
