/* EngramVault PWA Service Worker — v2.4.0 */

const CACHE_NAME = 'engramvault-shell-v2.4.0';

const PRECACHE_ASSETS = [
    '/',
    '/manifest.json',
    '/static/favicon.svg',
    '/static/icons/icon-192x192.png',
    '/static/icons/icon-512x512.png',
    '/static/icons/icon-maskable-512x512.png',
    '/static/icons/apple-touch-icon.png',
    '/static/css/gaskb.css',
    '/static/css/themes/themes.css'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(PRECACHE_ASSETS))
            .then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
            );
        }).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
    const request = event.request;
    const url = new URL(request.url);

    // Only process GET requests from same origin or CDNs
    if (request.method !== 'GET') {
        return;
    }

    // Dynamic backend and API routes: always go to network
    if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/mcp')) {
        return;
    }

    // Navigation requests (HTML documents)
    if (request.mode === 'navigate') {
        event.respondWith(
            fetch(request).catch(() => {
                return caches.match('/');
            })
        );
        return;
    }

    // Static assets: Stale-While-Revalidate
    if (url.pathname.startsWith('/static/') || url.pathname === '/manifest.json') {
        event.respondWith(
            caches.open(CACHE_NAME).then((cache) => {
                return cache.match(request).then((cachedResponse) => {
                    const fetchPromise = fetch(request).then((networkResponse) => {
                        if (networkResponse && networkResponse.status === 200) {
                            cache.put(request, networkResponse.clone());
                        }
                        return networkResponse;
                    }).catch(() => null);

                    return cachedResponse || fetchPromise;
                });
            })
        );
    }
});
