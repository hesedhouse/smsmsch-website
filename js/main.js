// ===== Navbar scroll effect =====
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
});

// ===== Mobile menu toggle =====
const navToggle = document.getElementById('navToggle');
const navMenu = document.getElementById('navMenu');

navToggle.addEventListener('click', () => {
  navMenu.classList.toggle('active');
});

navMenu.querySelectorAll('a').forEach(link => {
  link.addEventListener('click', () => {
    navMenu.classList.remove('active');
  });
});

// ===== Blog posts loading =====
async function loadBlogPosts() {
  const grid = document.getElementById('blogGrid');
  if (!grid) return;

  try {
    const res = await fetch('data/sermons.json');
    if (!res.ok) throw new Error('No data');
    const sermons = await res.json();

    if (!sermons.length) {
      grid.innerHTML = '<p style="grid-column:1/-1; text-align:center; color:#A0AEC0;">아직 등록된 말씀이 없습니다.</p>';
      return;
    }

    // Show latest 3 on homepage
    const latest = sermons.slice(0, 3);
    grid.innerHTML = latest.map(s => `
      <a href="blog/${s.slug}.html" class="blog-card">
        <p class="blog-card-date">${s.date}</p>
        <span class="blog-card-scripture">${s.scripture}</span>
        <h4>${s.title}</h4>
        <p>${s.summary}</p>
      </a>
    `).join('');

    if (sermons.length > 3) {
      document.getElementById('blogMore').style.display = 'block';
    }
  } catch {
    grid.innerHTML = '<p style="grid-column:1/-1; text-align:center; color:#A0AEC0;">설교 말씀이 곧 게시됩니다.</p>';
  }
}

document.addEventListener('DOMContentLoaded', loadBlogPosts);

// ===== Smooth reveal animation =====
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.about-card, .worship-card, .blog-card, .location-item, .youtube-channel-inner').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(24px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });
});
