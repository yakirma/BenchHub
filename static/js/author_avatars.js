/**
 * Git Author Avatar System
 * Generates consistent colored circles with initials for git authors
 */

// Store author colors to ensure consistency
const authorColors = {};

// Global mapping of author profiles (display names and avatar URLs)
// Injected via base.html/app.py context processor
window.AUTHOR_PROFILES = window.AUTHOR_PROFILES || {};

/**
 * Resolve the canonical profile for a username by following merge chains
 */
function resolveCanonicalProfile(username) {
    if (!username) return { name: '?', avatar: null };

    let visited = new Set();
    let currentName = username;
    let fallbackName = username;
    let avatar = null;

    while (currentName && window.AUTHOR_PROFILES[currentName]) {
        if (visited.has(currentName)) break; // Cycle detection
        visited.add(currentName);

        const profile = window.AUTHOR_PROFILES[currentName];
        if (profile.display_name) fallbackName = profile.display_name;
        if (profile.avatar_url) avatar = profile.avatar_url;

        if (profile.merged_into) {
            currentName = profile.merged_into;
        } else {
            break;
        }
    }

    return { name: fallbackName, avatar: avatar, username: currentName };
}

/**
 * Generate a consistent color for an author based on their name
 */
function getAuthorColor(authorName) {
    if (!authorName) return '#6c757d'; // Gray for unknown

    // Return cached color if exists
    if (authorColors[authorName]) {
        return authorColors[authorName];
    }

    // Generate a hash from the author name
    let hash = 0;
    for (let i = 0; i < authorName.length; i++) {
        hash = authorName.charCodeAt(i) + ((hash << 5) - hash);
    }

    // Convert hash to HSL color (varied hue, consistent saturation and lightness)
    const hue = Math.abs(hash % 360);
    const saturation = 65; // Good saturation for visibility
    const lightness = 50;  // Medium lightness for good contrast

    const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;

    // Cache the color
    authorColors[authorName] = color;

    return color;
}

/**
 * Extract initials from author name
 */
function getAuthorInitials(authorName) {
    if (!authorName) return '?';

    const profile = resolveCanonicalProfile(authorName);
    const nameToUse = profile.name;

    const parts = nameToUse.trim().split(/\s+/);

    if (parts.length === 1) {
        // Single name: take first 2 characters
        return parts[0].substring(0, 2).toUpperCase();
    } else {
        // Multiple names: take first letter of first and last name
        return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }
}

/**
 * Create an author avatar element
 */
function createAuthorAvatar(authorName, size = 43) {
    if (!authorName) {
        return `<div class="author-avatar" style="width: ${size}px; height: ${size}px; background-color: #6c757d; display: flex; align-items: center; justify-content: center; color: white; border-radius: 50%; font-family: sans-serif; font-weight: bold;" title="Unknown author">?</div>`;
    }

    const profile = resolveCanonicalProfile(authorName);
    const displayName = profile.name;
    const avatarUrl = profile.avatar;

    if (avatarUrl) {
        return `<div class="author-avatar" style="width: ${size}px; height: ${size}px; border-radius: 50%; overflow: hidden; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center;" title="${displayName}">
            <img src="${avatarUrl}" alt="${displayName}" style="width: 100%; height: 100%; object-fit: cover;">
        </div>`;
    }

    const initials = getAuthorInitials(authorName);
    const color = getAuthorColor(profile.username); // Use canonical name for consistent color
    const fontSize = Math.round(size * 0.4); // Font size is 40% of avatar size

    return `<div class="author-avatar" style="width: ${size}px; height: ${size}px; background-color: ${color}; display: flex; align-items: center; justify-content: center; color: white; border-radius: 50%; font-family: sans-serif; font-weight: bold; font-size: ${fontSize}px;" title="${displayName}">${initials}</div>`;
}
