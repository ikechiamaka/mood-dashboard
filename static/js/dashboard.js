// Role configuration and dynamic UI logic extracted from inline script
const roles = {
  super_admin: { id:'super_admin', name:'Super Administrator', description:'You have full access to all facilities and patients.', badgeClass:'super-admin-badge', badgeText:'Super Admin', userName:'Alex Johnson (Super Admin)', facilityAccess:'all', patientAccess:'all' },
  facility_admin: { id:'facility_admin', name:'Facility Administrator', description:'You have access to all patients in Facility 1.', badgeClass:'facility-admin-badge', badgeText:'Facility Admin', userName:'Sarah Miller (Facility 1 Admin)', facilityId:1, patientAccess:'facility' },
  staff: { id:'staff', name:'Staff Provider', description:'You have access only to your assigned patients in Facility 1.', badgeClass:'staff-badge', badgeText:'Staff Provider', userName:'James Wilson (Staff Provider)', facilityId:1, assignedPatientIds:[1,3,5] }
};
// Default to staff until server confirms role (prevents admin flicker)
let currentUser = roles.staff;
console.log('[dashboard] script loaded');
let currentPatient = null;
let currentUserProfile = null;
let checkinsCache = [];
let goalsCache = [];
let adminUiInitialized = false;
let sidebarOpen = false;
let csrfToken = null;
let drawerFocusCleanup = null;
let commandPaletteFocusCleanup = null;
let drawerReturnFocus = null;

function trapFocus(container){
  if(!container){
    return () => {};
  }
  const focusableSelector = 'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';
  const focusables = Array.from(container.querySelectorAll(focusableSelector))
    .filter(el => !el.hasAttribute('disabled') && !el.getAttribute('aria-hidden'));
  if(!focusables.length){
    return () => {};
  }
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  const handleKey = (evt) => {
    if(evt.key !== 'Tab'){
      return;
    }
    if(evt.shiftKey && document.activeElement === first){
      evt.preventDefault();
      last.focus();
    }else if(!evt.shiftKey && document.activeElement === last){
      evt.preventDefault();
      first.focus();
    }
  };
  container.addEventListener('keydown', handleKey);
  return () => container.removeEventListener('keydown', handleKey);
}

function buildAvatarUrl(_name){
  return '/static/pic.jpg';
}

function isLocalAssetUrl(value){
  if(!value || typeof value !== 'string'){
    return false;
  }
  return value.startsWith('/') && !value.startsWith('//');
}

function refreshUserAvatar(profile){
  const name = (profile && (profile.name || profile.email)) || 'User';
  const fallback = buildAvatarUrl(name);
  const src = profile && isLocalAssetUrl(profile.avatar_url) ? profile.avatar_url : fallback;
  const navAvatar = document.getElementById('currentUserAvatar');
  if(navAvatar){
    navAvatar.src = src;
    navAvatar.alt = `${name} avatar`;
  }
  const profileAvatar = document.getElementById('profileAvatar');
  if(profileAvatar){
    profileAvatar.src = src;
    profileAvatar.alt = `${name} avatar`;
  }
}

function getToastContainer(){
  let container = document.getElementById('toastContainer');
  if(!container){
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.setAttribute('aria-live', 'polite');
    container.setAttribute('aria-atomic', 'true');
    document.body.appendChild(container);
  }
  return container;
}

function showToast({ title = 'Notice', message = '', variant = 'success', delay = 4000 } = {}){
  if(typeof bootstrap === 'undefined' || typeof bootstrap.Toast === 'undefined'){
    console.log(`[toast:${variant}] ${title} - ${message}`);
    return;
  }
  const container = getToastContainer();
  const wrapper = document.createElement('div');
  wrapper.className = `toast align-items-start ${variant === 'error' ? 'toast-error' : 'toast-success'}`;
  wrapper.setAttribute('role', 'status');
  wrapper.setAttribute('aria-live', 'polite');
  wrapper.setAttribute('aria-atomic', 'true');
  wrapper.innerHTML = `
    <div class="toast-header">
      <strong class="me-auto">${escapeHtml(title)}</strong>
      <button type="button" class="btn-close ms-2 mb-1" data-bs-dismiss="toast" aria-label="Close"></button>
    </div>
    <div class="toast-body">${escapeHtml(message)}</div>
  `;
  container.appendChild(wrapper);
  const toast = bootstrap.Toast.getOrCreateInstance(wrapper, { delay });
  toast.show();
  wrapper.addEventListener('hidden.bs.toast', () => {
    if(wrapper.parentElement){
      wrapper.parentElement.removeChild(wrapper);
    }
  });
}

function updateRiskBadge(value){
  const badge = document.getElementById('patientRiskBadge');
  if(!badge) return;
  const normalized = (value || '').toString();
  const classes = {
    High: 'badge bg-danger text-white',
    Moderate: 'badge bg-warning text-dark',
    Low: 'badge bg-success text-white'
  };
  badge.className = classes[normalized] || 'badge bg-secondary text-white';
  badge.textContent = `Risk: ${normalized ? normalized : '--'}`;
}

function switchRole(roleId){ currentUser = roles[roleId]; updateUIForCurrentRole(); }
function updateUIForCurrentRole(){
  const roleBadge = document.getElementById('roleBadge');
  if(roleBadge){
    const classes = ['role-badge', 'role-badge-gap'];
    if(currentUser.badgeClass){
      classes.push(currentUser.badgeClass);
    }
    roleBadge.className = classes.join(' ');
    roleBadge.textContent = currentUser.badgeText;
  }
  const roleDesc = document.getElementById('roleDescription'); if(roleDesc) roleDesc.textContent = currentUser.description;
  const userNameEl = document.getElementById('currentUserName'); if(userNameEl) userNameEl.textContent = currentUser.userName;
  const staffWarning = document.getElementById('staffWarning'); if(staffWarning){ currentUser.id==='staff'?staffWarning.classList.remove('d-none'):staffWarning.classList.add('d-none'); }
  const facilityMgmt = document.getElementById('facilityMgmt');
  if(facilityMgmt){ facilityMgmt.classList.add('d-none'); facilityMgmt.style.display='none'; }
  const facilityNav = document.getElementById('facilityMgmtNav');
  if(facilityNav){
    const isAdmin = currentUser.id === 'super_admin' || currentUser.id === 'facility_admin';
    facilityNav.style.display = isAdmin ? '' : 'none';
    facilityNav.classList.toggle('d-none', !isAdmin);
  }
  const userMgmt = document.getElementById('userMgmt');
  const userNav = document.getElementById('userMgmtNav');
  if(userMgmt){ userMgmt.classList.add('d-none'); userMgmt.style.display='none'; }
  if(userNav){ userNav.style.display='none'; userNav.classList.add('d-none'); }
  const adminNav = document.getElementById('adminNav');
  if(adminNav){
    const isAdmin = currentUser.id === 'super_admin' || currentUser.id === 'facility_admin';
    adminNav.style.display = isAdmin ? '' : 'none';
  }
  filterDataBasedOnRole();
}
function filterDataBasedOnRole(){
  let dataStatus = '';
  switch(currentUser.id){
    case 'super_admin':
      dataStatus = 'Monitoring every facility across the network.';
      break;
    case 'facility_admin':
      dataStatus = currentUser.facilityId ? `Focused on Facility ${currentUser.facilityId} and all associated patients.` : 'Focused on your facility cohort.';
      break;
    case 'staff':
      dataStatus = currentUser.assignedPatientIds && currentUser.assignedPatientIds.length
        ? `Tracking assigned patients: ${currentUser.assignedPatientIds.join(', ')}.`
        : 'Ready to monitor your assigned caseload.';
      break;
    default:
      dataStatus = 'Calibrating access scope...';
  }
  const scope = document.getElementById('heroScopeText');
  if(scope){
    scope.textContent = dataStatus;
  }
  console.log('Data filtering applied:', dataStatus);
}
// Load current user (role) from server and map to our descriptors
async function initUser(){
  console.log('[dashboard] initUser start');
  try{
    const me = await apiGet('/api/me');
    console.log('[dashboard] /api/me result', me);
    if(!me) return;
    const base = roles[me.role] || roles.staff;
    currentUserProfile = me;
    refreshUserAvatar(me);
    csrfToken = me.csrf_token || null;
    currentUser = {
      ...base,
      userName: me.name || base.userName,
      facilityId: me.facility_id ?? base.facilityId,
      assignedPatientIds: Array.isArray(me.assigned_patient_ids) ? me.assigned_patient_ids : (base.assignedPatientIds||[])
    };
    // Hide demo role switcher once server-assigned role is loaded
    const switcher = document.getElementById('roleSwitcherContainer'); if(switcher) switcher.style.display='none';
    updateUIForCurrentRole();
  }catch(e){ console.error('[dashboard] loadDashboardData error', e); }
}
// Theme handling
function initTheme(){ const themeSwitch=document.getElementById('themeSwitch'); const themeSelect=document.getElementById('themeSelect'); if(localStorage.theme==='dark'){ document.body.classList.add('dark-theme'); if(themeSwitch) themeSwitch.checked=true; if(themeSelect) themeSelect.value='dark'; }
  if(themeSwitch){ themeSwitch.onchange=e=>{ const dark=e.target.checked; document.body.classList.toggle('dark-theme', dark); if(themeSelect) themeSelect.value=dark?'dark':'light'; localStorage.theme=dark?'dark':'light'; }; }
  if(themeSelect){ themeSelect.onchange=e=>{ const dark=e.target.value==='dark'; document.body.classList.toggle('dark-theme', dark); if(themeSwitch) themeSwitch.checked=dark; localStorage.theme=e.target.value; }; }
}
// Avatar upload feedback
function initAvatarUpload(){
  const avatarUpload = document.getElementById('avatarUpload');
  if(!avatarUpload){
    return;
  }
  avatarUpload.onchange = async () => {
    const file = avatarUpload.files && avatarUpload.files[0];
    if(!file){
      return;
    }
    try{
      await uploadProfileAvatar(file);
      showToast({ title:'Profile updated', message:'Avatar updated successfully.' });
    }catch(e){
      console.error('[dashboard] avatar upload error', e);
      showToast({ title:'Upload failed', message:'Unable to update avatar right now.', variant:'error' });
    }finally{
      avatarUpload.value = '';
    }
  };
}

async function uploadProfileAvatar(file){
  const formData = new FormData();
  formData.append('avatar', file);
  const resp = await fetch('/api/profile/avatar', {
    method:'POST',
    headers: withCsrf(),
    body: formData,
    credentials:'same-origin'
  });
  if(resp.status === 401){
    window.location.href = '/login';
    return null;
  }
  if(!resp.ok){
    throw new Error(`Avatar upload failed: ${resp.status}`);
  }
  try{
    const data = await resp.json();
    if(data && data.avatar_url){
      currentUserProfile = { ...(currentUserProfile || {}), avatar_url: data.avatar_url };
      refreshUserAvatar(currentUserProfile);
    }
    return data;
  }catch(_){
    return null;
  }
}
// Charts
// Keep global chart references for dynamic updates
const charts = { healthDonut:null, mood:null, movement:null };
let selectedPatientId = null;
function initCharts(){
  if(typeof Chart==='undefined') return;
  const donutEl = document.getElementById('healthDonut');
  if(donutEl){
    charts.healthDonut = new Chart(donutEl, {
      type:'doughnut',
      data:{ labels:['Mood','Activity','Sleep','Environment'], datasets:[{ data:[35,25,20,20], backgroundColor:['#4e73df','#1cc88a','#36b9cc','#f6c23e'], borderWidth:0 }]},
      options:{ cutout:'75%', plugins:{ legend:{display:false}, tooltip:{ callbacks:{ label: ctx=>`${ctx.label}: ${ctx.parsed}%` } } }, responsive:true, maintainAspectRatio:false }
    });
  }
  const moodCtx=document.getElementById('moodChart');
  if(moodCtx){
    charts.mood = new Chart(moodCtx.getContext('2d'), {
      type:'line',
      data:{ labels:['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], datasets:[{ label:'Mood Level', data:[3.2,4.1,3.8,4.5,4.8,5.2,4.9], borderColor:'#4e73df', backgroundColor:'rgba(78,115,223,0.05)', tension:0.4, fill:true, pointBackgroundColor:'#fff', pointBorderColor:'#4e73df', pointBorderWidth:2, pointRadius:4 }]},
      options:{ responsive:true, maintainAspectRatio:false, scales:{ y:{ min:1, max:6, ticks:{ callback:value=>['','Sad','Low','Neutral','Fair','Good','Happy'][value] } } }, plugins:{ legend:{display:false}, tooltip:{ callbacks:{ label:ctx=>`Mood: ${['','Sad','Low','Neutral','Fair','Good','Happy'][ctx.parsed.y]}` } } } }
    });
  }
  const moveCtx=document.getElementById('moveChart');
  if(moveCtx){
    charts.movement = new Chart(moveCtx.getContext('2d'), {
      type:'bar',
      data:{ labels:['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], datasets:[{ label:'Movement', data:[7.2,8.5,6.8,9.2,8.4,10.1,7.9], backgroundColor:'#1cc88a', borderRadius:5 }]},
      options:{ responsive:true, maintainAspectRatio:false, scales:{ y:{ beginAtZero:true, title:{ display:true, text:'Movement'} } }, plugins:{ legend:{display:false} } }
    });
  }
}

// API helpers and dynamic population
function withQuery(path, params){
  const url = new URL(path, window.location.origin);
  Object.entries(params||{}).forEach(([k,v])=>{ if(v!==undefined && v!==null && v!=='') url.searchParams.set(k, v); });
  return url.pathname + (url.search ? url.search : '');
}

function withCsrf(headers){
  const merged = { ...(headers || {}) };
  if(csrfToken){
    merged['X-CSRF-Token'] = csrfToken;
  }
  return merged;
}

async function apiGet(path, params){
  const url = params? withQuery(path, params) : path;
  const r = await fetch(url, { credentials:'same-origin' });
  if(r.status === 401) { window.location.href = '/login'; return null; }
  if(!r.ok) throw new Error(`GET ${path} failed: ${r.status}`);
  return r.json();
}

async function apiPost(path, payload, params){
  const url = params? withQuery(path, params) : path;
  const r = await fetch(url, { method:'POST', headers: withCsrf({ 'Content-Type':'application/json' }), credentials:'same-origin', body: JSON.stringify(payload) });
  if(r.status === 401) { window.location.href = '/login'; return null; }
  if(!r.ok) throw new Error(`POST ${path} failed: ${r.status}`);
  return r.json();
}

async function apiPatch(path, payload, params){
  const url = params ? withQuery(path, params) : path;
  const r = await fetch(url, { method:'PATCH', headers: withCsrf({ 'Content-Type':'application/json' }), credentials:'same-origin', body: JSON.stringify(payload) });
  if(r.status === 401){ window.location.href = '/login'; return null; }
  if(!r.ok) throw new Error(`PATCH ${path} failed: ${r.status}`);
  return r.json();
}

async function apiDelete(path, params){
  const url = params ? withQuery(path, params) : path;
  const r = await fetch(url, { method:'DELETE', headers: withCsrf(), credentials:'same-origin' });
  if(r.status === 401){ window.location.href = '/login'; return null; }
  if(r.status === 204) return null;
  if(!r.ok) throw new Error(`DELETE ${path} failed: ${r.status}`);
  try{
    return await r.json();
  }catch(_){
    return null;
  }
}

function setText(id, value){ const el = document.getElementById(id); if(el) el.textContent = value; }

function moodNumberToLabel(n){
  const labels = ['','Sad','Low','Neutral','Fair','Good','Happy'];
  if(typeof n !== 'number' || n < 1 || n > 6) return 'Unknown';
  return labels[n];
}

function formatDateTime(value){
  if(!value) return '--';
  const date = value instanceof Date ? value : new Date(value);
  if(Number.isNaN(date.getTime())) return '--';
  return date.toLocaleString(undefined, { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' });
}

function formatRelativeTime(value){
  if(!value) return '';
  const date = value instanceof Date ? value : new Date(value);
  if(Number.isNaN(date.getTime())) return '';
  const diff = Date.now() - date.getTime();
  const minutes = Math.round(diff / 60000);
  if(minutes < 1) return 'moments ago';
  if(minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if(hours < 24) return `${hours} hr${hours > 1 ? 's' : ''} ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days > 1 ? 's' : ''} ago`;
}

function updateDonutFromSlices(slices){
  if(!charts.healthDonut) return;
  const mood = Math.max(0, Math.min(100, slices.mood || 0));
  const activity = Math.max(0, Math.min(100, slices.movement || 0));
  const environment = Math.max(0, Math.min(100, slices.light || 0));
  const used = mood + activity + environment;
  const sleep = Math.max(0, 100 - used);
  charts.healthDonut.data.datasets[0].data = [mood, activity, sleep, environment];
  charts.healthDonut.update();
  const center = document.getElementById('healthDonutCenter');
  if(center){
    center.textContent = typeof slices.overallScore === 'number' ? `${Math.round(slices.overallScore)}%` : '--';
  }
  const overallBadge = document.getElementById('patientOverallScoreCard');
  if(overallBadge){
    overallBadge.textContent = typeof slices.overallScore === 'number' ? `${Math.round(slices.overallScore)}%` : '--';
  }
  if(typeof updateHealthStatusOverview === 'function') updateHealthStatusOverview(slices);
}

function updateTrendCharts(chartsPayload){
  try{
    if(charts.mood && Array.isArray(chartsPayload.mood)){
      const labels = chartsPayload.mood.map(p=>p.date);
      const data = chartsPayload.mood.map(p=>p.value ?? null);
      charts.mood.data.labels = labels;
      charts.mood.data.datasets[0].data = data;
      charts.mood.update();
    }
    if(charts.movement && Array.isArray(chartsPayload.movement)){
      const labels = chartsPayload.movement.map(p=>p.date);
      const data = chartsPayload.movement.map(p=>p.value ?? 0);
      charts.movement.data.labels = labels;
      charts.movement.data.datasets[0].data = data;
      charts.movement.update();
    }
  }catch(e){ console.warn('Chart update failed', e); }
}

function updateHealthStatusOverview(slices){
  try{
    const mood = Math.max(0, Math.min(100, slices.mood || 0));
    const activity = Math.max(0, Math.min(100, slices.movement || 0));
    const overall = Math.max(0, Math.min(100, slices.overallScore || 0));
    const setBar = (id, pct)=>{ const el=document.getElementById(id); if(el) el.style.width = `${Math.round(pct)}%`; };
    setBar('mentalProgress', mood);
    setBar('activityProgress', activity);
    setBar('physicalProgress', overall);
    const rank = (v)=> v>=75? {cls:'text-success', txt:'Strong'} : v>=45? {cls:'text-warning', txt:'Moderate'} : {cls:'text-danger', txt:'Low'};
    const upd = (id, v)=>{ const el=document.getElementById(id); if(!el) return; const r=rank(v); el.className = r.cls; el.textContent = r.txt; };
    upd('mentalStatus', mood);
    upd('activityStatus', activity);
    upd('physicalStatus', overall);
  }catch(e){ /* ignore */ }
}

function applyDashboardData(data){
  if(!data){
    return;
  }
  const s = data.summary || {};
  setText('avgTempValue', (s.avgTemp != null && !isNaN(s.avgTemp)) ? `${s.avgTemp}\u00B0F` : '--');
  setText('avgHumidityValue', (s.avgHumidity != null && !isNaN(s.avgHumidity)) ? `${s.avgHumidity}%` : '--');
  const aq = document.getElementById('airQualityBadge');
  if(aq && s.currentAirQuality){
    aq.textContent = s.currentAirQuality;
    aq.className = 'badge-indicator ' + (s.currentAirQuality==='Good' ? 'bg-success text-white' : 'bg-warning');
  }
  setText('patientMovementStatus', s.movementStatus || '--');
  setText('patientEnvironmentStatus', s.currentAirQuality || '--');
  setText('patientDrawerMovement', s.movementStatus || '--');
  setText('patientDrawerEnvironment', s.currentAirQuality || '--');
  setText('healthTrendLabel', s.movementStatus || 'Live');
  setText('healthStatusLabel', s.currentAirQuality || 'Optimal');
  if(typeof s.overallScore === 'number'){
    const overallDisplay = `${Math.round(s.overallScore)}%`;
    setText('patientOverallScore', overallDisplay);
    setText('patientOverallScoreCard', overallDisplay);
    setText('patientDrawerOverall', overallDisplay);
    setText('healthDonutCenter', overallDisplay);
  } else {
    setText('patientOverallScore', '--');
    setText('patientOverallScoreCard', '--');
    setText('patientDrawerOverall', '--');
    setText('healthDonutCenter', '--');
  }
  const heroSummaryParts = [];
  if(currentPatient && currentPatient.name){
    heroSummaryParts.push(`Now viewing: ${currentPatient.name}`);
  }
  if(s.movementStatus){
    heroSummaryParts.push(`Movement ${s.movementStatus}`);
  }
  if(s.currentAirQuality){
    heroSummaryParts.push(`Environment ${s.currentAirQuality}`);
  }
  if(typeof s.overallScore === 'number'){
    heroSummaryParts.push(`${Math.round(s.overallScore)}% overall`);
  }
  if(heroSummaryParts.length){
    setText('heroPatientContext', heroSummaryParts.join(' | '));
  } else if(currentPatient && currentPatient.name){
    setText('heroPatientContext', `Monitoring live signals for ${currentPatient.name}`);
  }
  const found = patientsCache.find(p => String(p.id) === String(selectedPatientId));
  if(found){
    if(typeof s.overallScore === 'number') found.computed_score = s.overallScore;
    if(s.movementStatus) found.movement_label = s.movementStatus;
    if(s.currentAirQuality) found.environment_label = s.currentAirQuality;
    renderPatientList();
    updatePatientDrawer(found);
  }
  if(data.donutSlices) updateDonutFromSlices(data.donutSlices);
  if(data.charts) updateTrendCharts(data.charts);
}

async function loadDashboardData(dataOverride){
  try{
    const data = dataOverride || await apiGet('/api/dashboard_data', { patient_id: selectedPatientId });
    if(!data) return;
    applyDashboardData(data);
  }catch(e){ console.error('[dashboard] initUser error', e); }
}

function applyWeeklyInsights(info){
  if(info && info.narrative){
    setText('weeklyInsights', info.narrative);
  } else if(info === null){
    setText('weeklyInsights', 'No data available for the past week.');
  }
}

async function loadWeeklyInsights(infoOverride){
  try{
    const info = infoOverride || await apiGet('/api/weekly_insights', { patient_id: selectedPatientId });
    applyWeeklyInsights(info);
  }catch(e){ console.error('[dashboard] loadWeeklyInsights error', e); }
}

async function loadLatestAndPredict(latestOverride, predictedOverride){
  try{
    setText('patientLatestUpdate', '--');
    setText('patientLatestVitals', '--');
    setText('patientDrawerLastUpdate', '--');
    setText('patientDrawerVitals', '--');
    setText('patientDrawerMood', '--');
    setText('heroLastUpdated', '--');
    setText('heroVitalsContext', 'Sync devices to refresh vitals');
    setText('heroMoodValue', '--');
    setText('heroMoodContext', 'Awaiting model prediction');
    const useOverride = latestOverride !== undefined;
    const latest = useOverride ? latestOverride : await apiGet('/api/latest_reading', { patient_id: selectedPatientId });
    if(!latest){
      setText('lastUpdatedText', 'No recent readings');
      setText('patientDrawerLastUpdate', '--');
      setText('patientDrawerVitals', '--');
      setText('patientDrawerMood', '--');
      setText('predictedMoodValue', '--');
      setText('heroLastUpdated', 'No data');
      setText('heroVitalsContext', 'Sync devices to refresh vitals');
      setText('heroMoodValue', '--');
      setText('heroMoodContext', 'Awaiting model prediction');
      return;
    }
    const ts = latest.timestamp ? new Date(latest.timestamp) : null;
    const lastUpdatedDisplay = ts ? ts.toLocaleString() : 'Just now';
    setText('lastUpdatedText', ts ? `Updated at ${lastUpdatedDisplay}` : 'Updated just now');
    setText('patientLatestUpdate', lastUpdatedDisplay);
    setText('patientDrawerLastUpdate', lastUpdatedDisplay);
    setText('heroLastUpdated', ts ? ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : lastUpdatedDisplay);
    const vitalsParts = [];
    const toNumber = (value)=>{ const num = Number(value); return Number.isFinite(num) ? num : null; };
    const temp = toNumber(latest.Temperature);
    if(temp !== null) vitalsParts.push(`${Math.round(temp)}\\u00B0F`);
    const hum = toNumber(latest.Humidity);
    if(hum !== null) vitalsParts.push(`${Math.round(hum)}% RH`);
    const move = toNumber(latest.Ultrasonic);
    if(move !== null) vitalsParts.push(`${Math.round(move)} cm movement`);
    const light = toNumber(latest.BH1750FVI);
    if(light !== null) vitalsParts.push(`${Math.round(light)} lux`);
    const vitalsText = vitalsParts.length ? vitalsParts.join(' | ') : 'No sensor data';
    setText('patientLatestVitals', vitalsText);
    setText('patientDrawerVitals', vitalsText);
    setText('heroVitalsContext', vitalsText);
    let predictedMood = predictedOverride;
    if(predictedMood == null){
      const pred = await apiPost('/api/predict_mood', latest, { patient_id: selectedPatientId });
      predictedMood = pred && typeof pred.predicted_mood === 'number' ? pred.predicted_mood : null;
    }
    if(predictedMood != null){
      const moodLabel = moodNumberToLabel(predictedMood);
      setText('predictedMoodValue', moodLabel);
      setText('patientDrawerMood', moodLabel);
      setText('heroMoodValue', moodLabel);
      setText('heroMoodContext', 'Updated from latest sensor packet');
    } else {
      setText('patientDrawerMood', '--');
      setText('heroMoodValue', '--');
      setText('heroMoodContext', 'Awaiting model prediction');
    }
  }catch(e){
    console.error(e);
    setText('lastUpdatedText', 'No recent readings');
    setText('patientLatestUpdate', '--');
    setText('patientLatestVitals', '--');
    setText('predictedMoodValue', '--');
    setText('heroLastUpdated', 'No data');
    setText('heroVitalsContext', 'Sync devices to refresh vitals');
    setText('heroMoodValue', '--');
    setText('heroMoodContext', 'Awaiting model prediction');
  }
}

async function loadPatientBundle(){
  if(!selectedPatientId){
    return;
  }
  try{
    const bundle = await apiGet('/api/patient_bundle', { patient_id: selectedPatientId });
    if(!bundle){
      return;
    }
    if(bundle.dashboard){
      applyDashboardData(bundle.dashboard);
    }
    if(bundle.weekly_insights !== undefined){
      applyWeeklyInsights(bundle.weekly_insights ? { narrative: bundle.weekly_insights } : null);
    }
    if(bundle.latest_reading !== undefined){
      await loadLatestAndPredict(bundle.latest_reading, bundle.predicted_mood);
    }
    if(Array.isArray(bundle.checkins)){
      checkinsCache = bundle.checkins;
      renderCheckins(bundle.checkins);
    }
    if(Array.isArray(bundle.goals)){
      goalsCache = bundle.goals;
      renderGoals(bundle.goals);
    }
    if(Array.isArray(bundle.journal_entries)){
      renderJournal(bundle.journal_entries);
    }
  }catch(e){
    console.error('[dashboard] loadPatientBundle error', e);
  }
}

async function loadStaff(){ try{ const list = await apiGet('/api/staff'); renderStaff(list); } catch(e){ console.error(e); } }
function renderStaff(items){
  const ul = document.getElementById('staffList'); if(!ul) return; ul.innerHTML='';
  (items||[]).forEach(s=>{
    const li = document.createElement('li'); li.className='list-group-item d-flex justify-content-between align-items-center';
    const span = document.createElement('span');
    const strong = document.createElement('strong');
    strong.textContent = s?.name || '';
    span.appendChild(strong);
    if(s?.id){
      span.appendChild(document.createTextNode(' | '));
      const code = document.createElement('code');
      code.textContent = s.id;
      span.appendChild(code);
    }
    if(s?.phone){
      span.appendChild(document.createTextNode(` | ${s.phone}`));
    }
    li.appendChild(span);
    ul.appendChild(li);
  });
}
async function loadBeds(){
  try{
    const items = await apiGet('/api/beds');
    renderBeds(items);
  }catch(e){ console.error('[dashboard] loadBeds failed', e); }
}

function renderBeds(items){
  const list = document.getElementById('bedsList');
  if(!list) return;
  list.innerHTML = '';
  (items || []).forEach(b => {
    const li = document.createElement('li');
    li.className = 'list-group-item d-flex justify-content-between align-items-center';
    li.textContent = [b.name, b.room ? `Room ${b.room}` : null, b.patient ? `Patient: ${b.patient}` : null].filter(Boolean).join(' | ');
    list.appendChild(li);
  });
}

async function addBed(){
  const nameInput = document.getElementById('bedName');
  const roomInput = document.getElementById('bedRoom');
  const patientInput = document.getElementById('bedPatient');
  const name = nameInput?.value?.trim();
  if(!name){ return; }
  const payload = {
    name,
    room: roomInput?.value?.trim() || '',
    patient: patientInput?.value?.trim() || ''
  };
  try{
    await apiPost('/api/beds', payload);
    if(nameInput) nameInput.value = '';
    if(roomInput) roomInput.value = '';
    if(patientInput) patientInput.value = '';
    loadBeds();
  }catch(e){ console.error('[dashboard] addBed failed', e); }
}

async function addStaff(){
  const name = document.getElementById('staffName')?.value.trim();
  const phone = document.getElementById('staffPhone')?.value.trim();
  if(!name || !phone) return;
  await apiPost('/api/staff', { name, phone });
  if(document.getElementById('staffName')) document.getElementById('staffName').value='';
  if(document.getElementById('staffPhone')) document.getElementById('staffPhone').value='';
  loadStaff();
}

async function loadShifts(){ try{ const list = await apiGet('/api/schedule'); renderShifts(list); } catch(e){ console.error(e); } }
function renderShifts(items){
  const ul = document.getElementById('shiftsList'); if(!ul) return; ul.innerHTML='';
  (items||[]).forEach(sh=>{
    const li = document.createElement('li'); li.className='list-group-item';
    const days = (sh.days||[]).join(',');
    const name = document.createElement('strong');
    name.textContent = sh?.name || '';
    li.appendChild(name);
    li.appendChild(document.createTextNode(` | ${sh?.start || ''}-${sh?.end || ''} | days [${days}]`));
    li.appendChild(document.createElement('br'));
    li.appendChild(document.createTextNode('staff: '));
    const staffIds = Array.isArray(sh?.staff_ids) ? sh.staff_ids : [];
    if(!staffIds.length){
      li.appendChild(document.createTextNode('none'));
    }else{
      staffIds.forEach((id, idx) => {
        const code = document.createElement('code');
        code.textContent = String(id);
        li.appendChild(code);
        if(idx < staffIds.length - 1){
          li.appendChild(document.createTextNode(', '));
        }
      });
    }
    ul.appendChild(li);
  });
}
async function addShift(){
  const name = document.getElementById('shiftName')?.value.trim();
  const start = document.getElementById('shiftStart')?.value.trim();
  const end = document.getElementById('shiftEnd')?.value.trim();
  const daysRaw = document.getElementById('shiftDays')?.value.trim();
  const idsRaw = document.getElementById('shiftStaffIds')?.value.trim();
  if(!name || !start || !end || !daysRaw) return;
  const days = daysRaw.split(',').map(s=>parseInt(s,10)).filter(n=>!isNaN(n));
  const staff_ids = idsRaw? idsRaw.split(',').map(s=>s.trim()).filter(Boolean) : [];
  await apiPost('/api/schedule', { name, start, end, days, staff_ids });
  ['shiftName','shiftStart','shiftEnd','shiftDays','shiftStaffIds'].forEach(id=>{ const el=document.getElementById(id); if(el) el.value=''; });
  loadShifts();
}

function initFacilityUi(){
  if(!currentUser || (currentUser.id !== 'super_admin' && currentUser.id !== 'facility_admin')){
    return;
  }
  const addBedBtn = document.getElementById('addBedBtn');
  if(addBedBtn && !addBedBtn.dataset.bound){
    addBedBtn.dataset.bound = '1';
    addBedBtn.addEventListener('click', addBed);
  }
  const addStaffBtn = document.getElementById('addStaffBtn');
  if(addStaffBtn && !addStaffBtn.dataset.bound){
    addStaffBtn.dataset.bound = '1';
    addStaffBtn.addEventListener('click', addStaff);
  }
  const addShiftBtn = document.getElementById('addShiftBtn');
  if(addShiftBtn && !addShiftBtn.dataset.bound){
    addShiftBtn.dataset.bound = '1';
    addShiftBtn.addEventListener('click', addShift);
  }
  loadBeds(); loadStaff(); loadShifts();
}

function initSidebarNav(){
  const links = document.querySelectorAll('.sidebar .sidebar-link');
  links.forEach(a=>{
    a.addEventListener('click', (e)=>{
      const href = a.getAttribute('href') || '';
      // Active state
      document.querySelectorAll('.sidebar .sidebar-link').forEach(el=>el.classList.remove('active'));
      a.classList.add('active');
      // Smooth scroll to anchors
      if(href.startsWith('#') && href.length>1){
        const target = document.querySelector(href);
        if(target){ e.preventDefault(); target.scrollIntoView({ behavior:'smooth', block:'start' }); }
        // Nudge charts to resize after navigation
        if(href === '#trendsSection'){
          setTimeout(()=>{ try{ charts.mood && charts.mood.resize(); charts.movement && charts.movement.resize(); }catch(_){} }, 350);
        } else if(href === '#healthMetrics'){
          setTimeout(()=>{ try{ charts.healthDonut && charts.healthDonut.resize(); }catch(_){} }, 350);
        }
      }
    });
  });
}

function initSidebarToggle(){
  const toggle = document.getElementById('sidebarToggle');
  const overlay = document.getElementById('sidebarOverlay');
  const closeSidebar = () => {
    document.body.classList.remove('sidebar-open');
    sidebarOpen = false;
    if(toggle){
      toggle.setAttribute('aria-expanded', 'false');
    }
  };
  const openSidebar = () => {
    document.body.classList.add('sidebar-open');
    sidebarOpen = true;
    if(toggle){
      toggle.setAttribute('aria-expanded', 'true');
    }
  };
  if(toggle){
    toggle.addEventListener('click', () => {
      sidebarOpen ? closeSidebar() : openSidebar();
    });
  }
  if(overlay){
    overlay.addEventListener('click', closeSidebar);
  }
  const navLinks = document.querySelectorAll('.sidebar .sidebar-link');
  navLinks.forEach(link => link.addEventListener('click', () => {
    if(window.innerWidth <= 992){
      closeSidebar();
    }
  }));
  window.addEventListener('resize', () => {
    if(window.innerWidth > 992){
      closeSidebar();
    }
  });
}

// Override renderers to use clean ASCII separators for reliability on all platforms
// (avoids encoding issues seen with special bullet characters)
function renderBeds(items){
  const ul = document.getElementById('bedsList'); if(!ul) return; ul.innerHTML='';
  (items||[]).forEach(b=>{
    const li = document.createElement('li'); li.className='list-group-item d-flex justify-content-between align-items-center';
    const meta = [b.name, b.room?`Room ${b.room}`:null, b.patient?`Patient: ${b.patient}`:null].filter(Boolean).join(' | ');
    li.textContent = `${meta}`; ul.appendChild(li);
  });
}
function renderStaff(items){
  const ul = document.getElementById('staffList'); if(!ul) return; ul.innerHTML='';
  (items||[]).forEach(s=>{
    const li = document.createElement('li'); li.className='list-group-item d-flex justify-content-between align-items-center';
    const span = document.createElement('span');
    const strong = document.createElement('strong');
    strong.textContent = s?.name || '';
    span.appendChild(strong);
    if(s?.id){
      span.appendChild(document.createTextNode(' | '));
      const code = document.createElement('code');
      code.textContent = s.id;
      span.appendChild(code);
    }
    if(s?.phone){
      span.appendChild(document.createTextNode(` | ${s.phone}`));
    }
    li.appendChild(span);
    ul.appendChild(li);
  });
}
function renderShifts(items){
  const ul = document.getElementById('shiftsList'); if(!ul) return; ul.innerHTML='';
  (items||[]).forEach(sh=>{
    const li = document.createElement('li'); li.className='list-group-item';
    const days = (sh.days||[]).join(',');
    const name = document.createElement('strong');
    name.textContent = sh?.name || '';
    li.appendChild(name);
    li.appendChild(document.createTextNode(` | ${sh?.start || ''}-${sh?.end || ''} | days [${days}]`));
    li.appendChild(document.createElement('br'));
    li.appendChild(document.createTextNode('staff: '));
    const staffIds = Array.isArray(sh?.staff_ids) ? sh.staff_ids : [];
    if(!staffIds.length){
      li.appendChild(document.createTextNode('none'));
    }else{
      staffIds.forEach((id, idx) => {
        const code = document.createElement('code');
        code.textContent = String(id);
        li.appendChild(code);
        if(idx < staffIds.length - 1){
          li.appendChild(document.createTextNode(', '));
        }
      });
    }
    ul.appendChild(li);
  });
}

// ----- User management (admins) -----
async function loadUsers(){ try{ const list = await apiGet('/api/users'); renderUsers(list); } catch(e){ console.error(e); } }
function renderUsers(items){
  const ul = document.getElementById('usersList'); if(!ul) return; ul.innerHTML='';
  (items||[]).forEach(u=>{
    const li = document.createElement('li'); li.className='list-group-item d-flex justify-content-between align-items-center';
    const left = document.createElement('span');
    const strong = document.createElement('strong');
    strong.textContent = u?.email || '';
    left.appendChild(strong);
    const roleText = u?.role ? ` ${u.role}` : '';
    const facilityText = u?.facility_id != null ? ` facility ${u.facility_id}` : '';
    if(roleText || facilityText){
      left.appendChild(document.createTextNode(`${roleText}${facilityText}`));
    }
    const right = document.createElement('div');
    const del = document.createElement('button'); del.className='btn btn-sm btn-outline-danger'; del.textContent='Delete'; del.onclick=async ()=>{ if(confirm('Delete user?')){ await apiDeleteUser(u.email); loadUsers(); } };
    right.appendChild(del);
    li.appendChild(left); li.appendChild(right); ul.appendChild(li);
  });
}
async function addUser(){
  const email = document.getElementById('userEmail')?.value.trim();
  const name = document.getElementById('userName')?.value.trim();
  const password = document.getElementById('userPassword')?.value.trim();
  const role = document.getElementById('userRole')?.value;
  const facilityIdRaw = document.getElementById('userFacilityId')?.value.trim();
  const assignedRaw = document.getElementById('userAssigned')?.value.trim();
  const facility_id = facilityIdRaw && /^\d+$/.test(facilityIdRaw) ? parseInt(facilityIdRaw,10) : undefined;
  const assigned_patient_ids = assignedRaw ? assignedRaw.split(',').map(s=>parseInt(s,10)).filter(n=>!isNaN(n)) : [];
  if(!email || !password) return;
  await apiPost('/api/users', { email, name, password, role, facility_id, assigned_patient_ids });
  ['userEmail','userName','userPassword','userAssigned'].forEach(id=>{ const el=document.getElementById(id); if(el) el.value=''; });
  loadUsers();
}
async function apiDeleteUser(email){
  const r = await fetch(`/api/users/${encodeURIComponent(email)}`, { method:'DELETE', headers: withCsrf(), credentials:'same-origin' });
  if(r.status === 401){ window.location.href='/login'; return; }
  if(!r.ok){ console.error('DELETE /api/users failed'); }
}
function initAdminUi(){
  if(!currentUser || (currentUser.id !== 'super_admin' && currentUser.id !== 'facility_admin')){
    return;
  }
  if(!adminUiInitialized){
    adminUiInitialized = true;
    // Restrict role options for facility_admin (no super_admin creation)
    if(currentUser.id === 'facility_admin'){
      const roleSel = document.getElementById('userRole');
      if(roleSel){
        [...roleSel.options].forEach(o=>{ if(o.value==='super_admin') o.disabled = true; });
      }
    }
    const addUserBtn = document.getElementById('addUserBtn');
    if(addUserBtn && !addUserBtn.dataset.bound){
      addUserBtn.dataset.bound = '1';
      addUserBtn.addEventListener('click', addUser);
    }
  }
  loadUsers();
}

// ----- Journal -----
async function loadCheckins(){
  if(!selectedPatientId){
    renderCheckins([]);
    return;
  }
  try{
    const resp = await apiGet('/api/checkins', { patient_id: selectedPatientId });
    checkinsCache = Array.isArray(resp) ? resp : [];
    renderCheckins(checkinsCache);
  }catch(e){
    console.error('[dashboard] loadCheckins error', e);
    showToast({ title:'Check-ins unavailable', message:'Unable to load recent check-ins right now.', variant:'error' });
    renderCheckins([]);
  }
}

function renderCheckins(items){
  const list = document.getElementById('checkinsList');
  const empty = document.getElementById('checkinsEmpty');
  if(!list || !empty){
    return;
  }
  list.innerHTML = '';
  const data = Array.isArray(items) ? items.slice(0, 25) : [];
  if(!data.length){
    empty.textContent = selectedPatientId ? 'No check-ins recorded yet for this patient.' : 'Select a patient to view check-ins.';
    empty.classList.remove('d-none');
    return;
  }
  empty.classList.add('d-none');
  data.forEach(item => {
    const ts = item.timestamp ? new Date(item.timestamp) : null;
    const li = document.createElement('li');
    li.className = 'list-group-item checkin-item';
    const header = document.createElement('div');
    header.className = 'checkin-item-header';
    const moodLabel = moodNumberToLabel(Number(item.mood));
    const headerLeft = document.createElement('span');
    headerLeft.appendChild(document.createTextNode(moodLabel));
    headerLeft.appendChild(document.createTextNode(' '));
    const scoreBadge = document.createElement('span');
    scoreBadge.className = 'badge bg-light text-dark ms-2';
    scoreBadge.textContent = `Score ${item.mood ?? '--'}`;
    headerLeft.appendChild(scoreBadge);
    const headerRight = document.createElement('span');
    headerRight.className = 'text-muted small';
    headerRight.textContent = formatDateTime(ts);
    header.appendChild(headerLeft);
    header.appendChild(headerRight);
    const meta = document.createElement('div');
    meta.className = 'checkin-item-meta';
    const buildMeta = (iconClass, text) => {
      const span = document.createElement('span');
      const icon = document.createElement('i');
      icon.className = iconClass;
      span.appendChild(icon);
      span.appendChild(document.createTextNode(text));
      return span;
    };
    const stressRaw = (item.stress || 'Not captured').toString();
    const stressText = stressRaw ? stressRaw.replace(/^\w/, c=>c.toUpperCase()) : 'Not captured';
    meta.appendChild(buildMeta('fas fa-user-circle me-1', item.user || 'Unknown'));
    meta.appendChild(buildMeta('fas fa-bolt me-1', stressText));
    meta.appendChild(buildMeta('far fa-clock me-1', formatRelativeTime(ts)));
    li.appendChild(header);
    li.appendChild(meta);
    if(item.notes){
      const notes = document.createElement('div');
      notes.className = 'checkin-item-notes';
      notes.textContent = item.notes;
      li.appendChild(notes);
    }
    list.appendChild(li);
  });
}

async function submitCheckin(evt){
  evt.preventDefault();
  if(!selectedPatientId){
    showToast({ title:'Select a patient', message:'Choose a patient before recording a check-in.', variant:'error' });
    return;
  }
  const form = evt.target;
  const moodEl = document.getElementById('checkinMood');
  const stressEl = document.getElementById('checkinStress');
  const notesEl = document.getElementById('checkinNotes');
  const moodVal = parseInt(moodEl?.value ?? '', 10);
  if(!Number.isInteger(moodVal)){
    showToast({ title:'Mood required', message:'Please choose a mood score.', variant:'error' });
    moodEl?.focus();
    return;
  }
  const stressVal = (stressEl?.value || '').trim();
  const notesVal = (notesEl?.value || '').trim();
  const payload = {
    timestamp: new Date().toISOString(),
    mood: moodVal,
    stress: stressVal || null,
    notes: notesVal || null,
    patient_id: selectedPatientId
  };
  try{
    await apiPost('/api/checkins', payload, { patient_id: selectedPatientId });
    showToast({ title:'Check-in saved', message:'Mood check-in recorded successfully.' });
    form.reset();
    loadPatientBundle();
  }catch(e){
    console.error('[dashboard] submitCheckin error', e);
    showToast({ title:'Save failed', message:'We could not save this check-in. Please try again.', variant:'error' });
  }
}

async function loadGoals(){
  if(!selectedPatientId){
    renderGoals([]);
    return;
  }
  try{
    const resp = await apiGet('/api/goals', { patient_id: selectedPatientId });
    goalsCache = Array.isArray(resp) ? resp : [];
    renderGoals(goalsCache);
  }catch(e){
    console.error('[dashboard] loadGoals error', e);
    showToast({ title:'Goals unavailable', message:'Unable to load care goals right now.', variant:'error' });
    renderGoals([]);
  }
}

function renderGoals(items){
  const list = document.getElementById('goalsList');
  const empty = document.getElementById('goalsEmpty');
  if(!list || !empty){
    return;
  }
  list.innerHTML = '';
  const data = Array.isArray(items) ? items : [];
  if(!data.length){
    empty.textContent = selectedPatientId ? 'No goals created yet.' : 'Select a patient to view goals.';
    empty.classList.remove('d-none');
    return;
  }
  empty.classList.add('d-none');
  data.forEach(goal => {
    const li = document.createElement('li');
    li.className = 'list-group-item goals-list-item';
    const info = document.createElement('div');
    const titleEl = document.createElement('div');
    titleEl.className = 'fw-semibold';
    titleEl.textContent = goal.title || 'Untitled goal';
    info.appendChild(titleEl);
    const meta = document.createElement('div');
    meta.className = 'goal-meta';
    const statusBadge = document.createElement('span');
    const isCompleted = goal.status === 'completed';
    statusBadge.className = `badge ${isCompleted ? 'bg-success text-white' : 'bg-primary text-white'}`;
    statusBadge.textContent = isCompleted ? 'Completed' : 'Active';
    meta.appendChild(statusBadge);
    const dueSpan = document.createElement('span');
    const dueText = (() => {
      if(!goal.due_date){
        return 'No due date';
      }
      const due = new Date(goal.due_date);
      if(Number.isNaN(due.getTime())){
        return 'No due date';
      }
      return due.toLocaleDateString(undefined, { month:'short', day:'numeric', year:'numeric' });
    })();
    const dueIcon = document.createElement('i');
    dueIcon.className = 'far fa-calendar-alt me-1';
    dueSpan.appendChild(dueIcon);
    dueSpan.appendChild(document.createTextNode(dueText));
    meta.appendChild(dueSpan);
    const createdSpan = document.createElement('span');
    const createdIcon = document.createElement('i');
    createdIcon.className = 'far fa-clock me-1';
    createdSpan.appendChild(createdIcon);
    createdSpan.appendChild(document.createTextNode(formatRelativeTime(goal.created_at) || 'recently'));
    meta.appendChild(createdSpan);
    info.appendChild(meta);
    const actions = document.createElement('div');
    actions.className = 'btn-group btn-group-sm';
    const completeButton = document.createElement('button');
    completeButton.type = 'button';
    completeButton.className = `btn btn-outline-success${goal.status === 'completed' ? ' disabled' : ''}`;
    completeButton.innerHTML = '<i class="fas fa-check"></i>';
    completeButton.title = 'Mark complete';
    if(goal.status !== 'completed'){
      completeButton.addEventListener('click', () => markGoalComplete(goal.id));
    }
    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'btn btn-outline-danger';
    deleteButton.innerHTML = '<i class="fas fa-trash"></i>';
    deleteButton.title = 'Delete goal';
    deleteButton.addEventListener('click', () => deleteGoal(goal.id));
    actions.appendChild(completeButton);
    actions.appendChild(deleteButton);
    li.appendChild(info);
    li.appendChild(actions);
    list.appendChild(li);
  });
}

async function submitGoal(evt){
  evt.preventDefault();
  if(!selectedPatientId){
    showToast({ title:'Select a patient', message:'Choose a patient before adding a goal.', variant:'error' });
    return;
  }
  const form = evt.target;
  const title = (document.getElementById('goalTitle')?.value || '').trim();
  const dueDate = document.getElementById('goalDueDate')?.value || '';
  const notify = !!document.getElementById('goalNotify')?.checked;
  if(!title){
    showToast({ title:'Goal title required', message:'Enter a short description for the goal.', variant:'error' });
    return;
  }
  const payload = {
    title,
    patient_id: selectedPatientId,
    notify,
    due_date: dueDate || null
  };
  try{
    await apiPost('/api/goals', payload, { patient_id: selectedPatientId });
    showToast({ title:'Goal saved', message:'New care goal added.' });
    const modalEl = document.getElementById('goalModal');
    if(modalEl && typeof bootstrap !== 'undefined'){
      const modalInstance = bootstrap.Modal.getOrCreateInstance(modalEl);
      modalInstance.hide();
    }
    form.reset();
    loadGoals();
  }catch(e){
    console.error('[dashboard] submitGoal error', e);
    showToast({ title:'Save failed', message:'We could not add this goal. Please try again.', variant:'error' });
  }
}

async function markGoalComplete(goalId){
  if(!goalId){
    return;
  }
  try{
    await apiPatch(`/api/goals/${goalId}`, { status: 'completed' });
    showToast({ title:'Goal completed', message:'Goal marked as complete.' });
    loadGoals();
  }catch(e){
    console.error('[dashboard] markGoalComplete error', e);
    showToast({ title:'Update failed', message:'We could not update this goal.', variant:'error' });
  }
}

async function deleteGoal(goalId){
  if(!goalId){
    return;
  }
  try{
    await apiDelete(`/api/goals/${goalId}`);
    showToast({ title:'Goal removed', message:'Goal deleted successfully.' });
    loadGoals();
  }catch(e){
    console.error('[dashboard] deleteGoal error', e);
    showToast({ title:'Delete failed', message:'We could not delete this goal.', variant:'error' });
  }
}

async function loadJournal(){
  if(!selectedPatientId){
    renderJournal([]);
    return;
  }
  try{
    const entries = await apiGet('/api/journal_entries', { patient_id: selectedPatientId });
    renderJournal(entries || []);
  }catch(e){
    console.error('[dashboard] initUser error', e);
  }
}
function renderJournal(items){
  const list = document.getElementById('journalList');
  if(!list){
    return;
  }
  list.innerHTML = '';
  const entries = Array.isArray(items) ? items : [];
  entries.forEach(entry => {
    const li = document.createElement('li');
    li.className = 'list-group-item';
    const ts = entry.timestamp ? new Date(entry.timestamp) : null;
    const subtitle = ts && !Number.isNaN(ts.getTime()) ? ts.toLocaleString() : 'Journal entry';
    const titleRow = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = subtitle;
    titleRow.appendChild(strong);
    const body = document.createElement('div');
    body.textContent = (entry.text || '').toString();
    li.appendChild(titleRow);
    li.appendChild(body);
    list.appendChild(li);
  });
  updateDrawerJournalHighlight(entries);
}
function updateDrawerJournalHighlight(items){
  const notesEl = document.getElementById('patientDrawerNotes');
  if(!notesEl){
    return;
  }
  const entries = Array.isArray(items) ? items : [];
  if(!entries.length){
    notesEl.textContent = 'Add a journal entry to capture observations or interventions.';
    return;
  }
  const first = entries[0] || {};
  const ts = first.timestamp ? new Date(first.timestamp) : null;
  const metaParts = [];
  if(ts && !Number.isNaN(ts.getTime())){
    metaParts.push(ts.toLocaleString());
  }
  if(first.mood != null){
    const moodLabel = moodNumberToLabel(Number(first.mood));
    if(moodLabel){
      metaParts.push(`Mood: ${moodLabel}`);
    }
  }
  const prefix = metaParts.length ? `${metaParts.join(' | ')} - ` : '';
  const rawText = (first.text || '').toString().trim();
  const summary = rawText ? (rawText.length > 160 ? `${rawText.slice(0, 160)}...` : rawText) : 'No recent note yet.';
  notesEl.textContent = prefix + summary;
}
async function addJournal(){
  const textVal = document.getElementById('journalText')?.value.trim();
  const mood = parseInt(document.getElementById('journalMood')?.value, 10) || 3;
  if(!textVal){
    return;
  }
  if(!selectedPatientId){
    showToast({ title:'Select a patient', message:'Choose a patient before adding notes.', variant:'error' });
    return;
  }
  const payload = { timestamp: new Date().toISOString(), text: textVal, mood, patient_id: selectedPatientId };
  try{
    await apiPost('/api/journal_entries', payload, { patient_id: selectedPatientId });
    const box = document.getElementById('journalText');
    if(box){
      box.value = '';
    }
    showToast({ title:'Journal saved', message:'Journal entry added successfully.' });
    loadJournal();
  }catch(e){
    console.error('[dashboard] addJournal error', e);
    showToast({ title:'Save failed', message:'Unable to save the journal entry right now.', variant:'error' });
  }
}
// Bind journal button
document.addEventListener('click', (e)=>{
  if(e.target && e.target.id === 'addJournalBtn'){
    e.preventDefault();
    addJournal();
  }
});

// ----- Patients -----
const MAX_PATIENT_SUGGESTIONS = 7;
const PATIENT_ROW_HEIGHT = 80;
const PATIENT_GROUP_HEIGHT = 34;

let patientsCache = [];
let filteredPatients = [];
let patientRowMeta = [];
let patientRowsTotalHeight = 0;
let patientSearchTerm = '';
let currentSuggestions = [];
let suggestionIndex = -1;
let patientFilters = { facility: 'all', risk: 'all', wing: 'all' };
let patientFilterOptions = { facilities: [], riskLevels: [], wings: [] };
let patientGroupMode = 'facility';
let patientListRenderFrame = null;
let patientListResizeFrame = null;
let patientSearchInputEl;
let patientSuggestionsEl;
let patientListWrapperEl;
let patientQuickSelectEl;
let suppressQuickSelectChange = false;
let patientDrawerEl;
let patientDrawerPanelEl;
let patientDrawerOverlayEl;
let patientDrawerOpen = false;
const MAX_COMMAND_RESULTS = 12;
const commandPaletteActions = [
  {
    id: 'open-drawer',
    label: 'Open patient detail drawer',
    subtitle: 'Show the current patient summary',
    run: () => {
      if(currentPatient){
        updatePatientDrawer(currentPatient);
        openPatientDrawer();
      }else{
        showToast({ title:'No patient selected', message:'Choose a patient first.', variant:'error' });
      }
    }
  },
  {
    id: 'focus-journal',
    label: 'Add journal entry',
    subtitle: 'Focus the journal composer',
    run: () => {
      const textarea = document.getElementById('journalText');
      if(textarea){
        textarea.focus();
      }
    }
  },
  {
    id: 'record-checkin',
    label: 'Record a mood check-in',
    subtitle: 'Focus the check-in form',
    run: () => {
      const moodSelect = document.getElementById('checkinMood');
      if(moodSelect){
        moodSelect.focus();
      }
      const section = document.getElementById('checkinsSection');
      if(section){
        section.scrollIntoView({ behavior:'smooth', block:'start' });
      }
    }
  },
  {
    id: 'add-care-goal',
    label: 'Add care goal',
    subtitle: 'Open goal modal',
    run: () => {
      const modalEl = document.getElementById('goalModal');
      if(modalEl && typeof bootstrap !== 'undefined'){
        const instance = bootstrap.Modal.getOrCreateInstance(modalEl);
        instance.show();
      }
    }
  },
  {
    id: 'sync-devices',
    label: 'Sync wearable devices',
    subtitle: 'Trigger device sync now',
    run: () => {
      const btn = document.getElementById('syncWearable');
      if(btn){
        btn.click();
      }
    }
  },
  {
    id: 'refresh-dashboard',
    label: 'Refresh dashboard data',
    subtitle: 'Reload metrics for the current patient',
    run: () => {
      loadPatientBundle();
    }
  },
  {
    id: 'open-admin',
    label: 'Open admin workspace',
    subtitle: '/admin',
    run: () => {
      window.location.href = '/admin';
    }
  }
];
let commandPaletteEl;
let commandPaletteInput;
let commandPaletteList;
let commandPaletteMatches = [];
let commandPaletteIndex = -1;
let commandPaletteOpen = false;
let commandPaletteReturnFocus = null;


async function initPatientSelector(){
  patientSearchInputEl = document.getElementById('patientSearchInput');
  patientSuggestionsEl = document.getElementById('patientSearchSuggestions');
  patientListWrapperEl = document.querySelector('.patient-list-wrapper');
  ensurePatientQuickSelect();
  try{
    patientsCache = await apiGet('/api/patients');
  }catch(e){
    console.error('initPatientSelector failed', e);
    patientsCache = [];
  }
  computePatientFilterOptions();
  renderPatientFilterChips();
  bindPatientSearch();
  bindPatientListWrapper();
  bindGroupModeControl();

  const selectionChanged = filterPatients({ initial: true });
  renderPatientList({ resetScroll: true, force: true });

  if(!patientsCache.length){
    selectedPatientId = null;
    resetPatientCard();
    afterPatientChanged();
    return;
  }

  if(!selectedPatientId && filteredPatients.length){
    selectedPatientId = String(filteredPatients[0].id);
  }

  updatePatientDetails();
  if(selectionChanged){
    afterPatientChanged();
  }else{
    ensureSelectedVisible();
  }
}

function bindPatientListWrapper(){
  if(!patientListWrapperEl || patientListWrapperEl.dataset.bound){
    return;
  }
  patientListWrapperEl.dataset.bound = '1';
  patientListWrapperEl.addEventListener('scroll', () => {
    if(patientListRenderFrame){
      return;
    }
    patientListRenderFrame = requestAnimationFrame(() => {
      patientListRenderFrame = null;
      renderPatientList({ fromScroll: true });
    });
  }, { passive: true });
  window.addEventListener('resize', () => {
    if(patientListResizeFrame){
      cancelAnimationFrame(patientListResizeFrame);
    }
    patientListResizeFrame = requestAnimationFrame(() => {
      renderPatientList({ force: true });
      ensureSelectedVisible();
    });
  });
}

function bindGroupModeControl(){
  const select = document.getElementById('patientGroupMode');
  if(!select || select.dataset.bound){
    return;
  }
  select.dataset.bound = '1';
  select.value = patientGroupMode;
  select.addEventListener('change', () => {
    patientGroupMode = select.value || 'facility';
    buildVirtualRows();
    renderPatientList({ resetScroll: true, force: true });
    ensureSelectedVisible();
  });
}


function initPatientDrawer(){
  patientDrawerEl = document.getElementById('patientDetailDrawer');
  if(!patientDrawerEl){
    return;
  }
  patientDrawerPanelEl = patientDrawerEl.querySelector('.patient-drawer-panel');
  patientDrawerOverlayEl = document.getElementById('patientDrawerOverlay');
  const openBtn = document.getElementById('patientOpenDrawer');
  const closeBtn = document.getElementById('patientDrawerClose');
  if(openBtn && !openBtn.dataset.drawerBound){
    openBtn.dataset.drawerBound = '1';
    openBtn.addEventListener('click', () => {
      if(!currentPatient){
        showToast({ title:'No patient selected', message:'Choose a patient to open the drawer.', variant:'error' });
        return;
      }
      updatePatientDrawer(currentPatient);
      openPatientDrawer();
    });
  }
  if(closeBtn){
    closeBtn.addEventListener('click', closeCommandPalette);
    closeBtn.addEventListener('click', closePatientDrawer);
  }
  if(patientDrawerOverlayEl){
    patientDrawerOverlayEl.addEventListener('click', closePatientDrawer);
  }
  document.addEventListener('keydown', evt => {
    if(evt.key === 'Escape' && patientDrawerOpen){
      closePatientDrawer();
    }
  });
  resetPatientDrawer();
}

function openPatientDrawer(){
  if(!patientDrawerEl){
    return;
  }
  patientDrawerOpen = true;
  drawerReturnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  patientDrawerEl.classList.add('active');
  patientDrawerEl.setAttribute('aria-hidden', 'false');
  document.body.classList.add('drawer-open');
  if(drawerFocusCleanup){
    drawerFocusCleanup();
  }
  drawerFocusCleanup = trapFocus(patientDrawerEl);
  const focusTarget = patientDrawerEl.querySelector('[data-drawer-focus]') || patientDrawerEl.querySelector('button, [href], input, select, textarea');
  if(focusTarget){
    focusTarget.focus({ preventScroll: true });
  }
}

function closePatientDrawer(){
  if(!patientDrawerEl){
    return;
  }
  patientDrawerOpen = false;
  patientDrawerEl.classList.remove('active');
  patientDrawerEl.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('drawer-open');
  if(drawerFocusCleanup){
    drawerFocusCleanup();
    drawerFocusCleanup = null;
  }
  const focusTarget = drawerReturnFocus;
  drawerReturnFocus = null;
  if(focusTarget && typeof focusTarget.focus === 'function'){
    focusTarget.focus({ preventScroll: true });
  }
}

function resetPatientDrawer(){
  setText('patientDrawerName', 'No patient selected');
  setText('patientDrawerMeta', 'Choose a patient to see context and recent activity.');
  setText('patientDrawerSubtitle', 'Select a patient to open their detail drawer.');
  setText('patientDrawerCondition', '--');
  setText('patientDrawerCareFocus', '--');
  setText('patientDrawerAllergies', '--');
  setText('patientDrawerOverall', '--');
  setText('patientDrawerMovement', '--');
  setText('patientDrawerEnvironment', '--');
  setText('patientDrawerMood', '--');
  setText('patientDrawerVitals', '--');
  setText('patientDrawerLastUpdate', '--');
  setText('patientDrawerNotes', 'Add a journal entry to capture observations or interventions.');
  const facilityBadge = document.getElementById('patientDrawerFacility');
  if(facilityBadge){
    facilityBadge.textContent = 'Facility --';
    facilityBadge.className = 'badge bg-light text-muted';
  }
  const riskBadge = document.getElementById('patientDrawerRisk');
  if(riskBadge){
    riskBadge.className = 'badge patient-risk-badge risk-unknown';
    riskBadge.textContent = 'Risk: --';
  }
  const avatar = document.getElementById('patientDrawerAvatar');
  if(avatar){
    const fallback = avatar.dataset.defaultSrc || avatar.getAttribute('data-default-src') || avatar.src;
    avatar.src = fallback;
    avatar.alt = 'Patient avatar';
  }
}

function updatePatientDrawer(patient){
  if(!patient){
    resetPatientDrawer();
    return;
  }
  setText('patientDrawerName', patient.name || 'Unnamed patient');
  setText('patientDrawerLastUpdate', '--');
  setText('patientDrawerVitals', '--');
  setText('patientDrawerMood', '--');
  const metaBits = [];
  if(patient.id){
    metaBits.push(`ID ${String(patient.id).padStart(4, '0')}`);
  }
  if(patient.bed_id){
    metaBits.push(`Bed ${patient.bed_id}`);
  }
  if(patient.age != null){
    metaBits.push(`Age ${patient.age}`);
  }
  setText('patientDrawerMeta', metaBits.join(' | ') || 'Profile details pending');
  const subtitleParts = [];
  if(patient.facility_id){
    subtitleParts.push(`Facility ${patient.facility_id}`);
  }
  if(patient.bed_id){
    subtitleParts.push(`Bed ${patient.bed_id}`);
  }
  setText('patientDrawerSubtitle', subtitleParts.length ? subtitleParts.join(' | ') : 'Care snapshot');
  const facilityBadge = document.getElementById('patientDrawerFacility');
  if(facilityBadge){
    if(patient.facility_id){
      facilityBadge.textContent = `Facility ${patient.facility_id}`;
      facilityBadge.className = 'badge bg-primary-subtle text-primary';
    }else{
      facilityBadge.textContent = 'Facility Unassigned';
      facilityBadge.className = 'badge bg-light text-muted';
    }
  }
  const riskBadge = document.getElementById('patientDrawerRisk');
  if(riskBadge){
    riskBadge.className = 'badge patient-risk-badge ' + getRiskLevelClass(patient.risk_level);
    riskBadge.textContent = patient.risk_level ? `Risk: ${patient.risk_level}` : 'Risk: --';
  }
  setText('patientDrawerCondition', patient.primary_condition || 'No primary condition recorded');
  setText('patientDrawerCareFocus', patient.care_focus || 'Personalized plan pending');
  setText('patientDrawerAllergies', patient.allergies || 'No known allergies');
  if(typeof patient.computed_score === 'number'){
    setText('patientDrawerOverall', `${Math.round(patient.computed_score)}%`);
  }else{
    setText('patientDrawerOverall', '--');
  }
  setText('patientDrawerMovement', patient.movement_label || patient.movementStatus || '--');
  setText('patientDrawerEnvironment', patient.environment_label || patient.currentAirQuality || '--');
  const avatar = document.getElementById('patientDrawerAvatar');
  if(avatar){
    const fallback = avatar.dataset.defaultSrc || avatar.getAttribute('data-default-src') || avatar.src;
    const avatarUrl = (patient.avatar_url && isLocalAssetUrl(patient.avatar_url))
      ? patient.avatar_url
      : (patient.name ? buildAvatarUrl(patient.name) : null) || fallback;
    avatar.src = avatarUrl;
    avatar.alt = patient.name ? `${patient.name} avatar` : 'Patient avatar';
  }
}

function computePatientFilterOptions(){
  const facilityMap = new Map();
  const riskMap = new Map();
  const wingMap = new Map();
  let hasUnassignedWing = false;

  patientsCache.forEach(p => {
    if(p.facility_id != null){
      const id = String(p.facility_id);
      if(!facilityMap.has(id)){
        facilityMap.set(id, `Facility ${id}`);
      }
    }
    const rawRisk = (p.risk_level || '').toString().trim();
    if(rawRisk){
      const key = rawRisk.toLowerCase();
      if(!riskMap.has(key)){
        riskMap.set(key, titleCase(rawRisk));
      }
    }
    const wing = deriveBedWing(p.bed_id);
    if(wing === 'Unassigned'){
      hasUnassignedWing = true;
    }else if(wing){
      if(!wingMap.has(wing)){
        wingMap.set(wing, `Wing ${wing}`);
      }
    }
  });

  if(hasUnassignedWing){
    wingMap.set('Unassigned', 'Wing Unassigned');
  }

  patientFilterOptions.facilities = Array.from(facilityMap, ([value, label]) => ({ value, label }));
  patientFilterOptions.riskLevels = Array.from(riskMap, ([value, label]) => ({ value, label }));
  patientFilterOptions.wings = Array.from(wingMap, ([value, label]) => ({ value, label }));
}

function renderPatientFilterChips(){
  const container = document.getElementById('patientFilterChips');
  if(!container){
    return;
  }
  container.innerHTML = '';
  const frag = document.createDocumentFragment();

  frag.appendChild(createFilterChip('All Patients', 'all', 'all', 'chip-outline'));

  if(patientFilterOptions.facilities.length){
    frag.appendChild(createFilterLabel('Facility'));
    patientFilterOptions.facilities.forEach(opt => {
      frag.appendChild(createFilterChip(opt.label, 'facility', opt.value, 'chip-neutral'));
    });
  }

  if(patientFilterOptions.riskLevels.length){
    frag.appendChild(createFilterLabel('Risk Tier'));
    patientFilterOptions.riskLevels.forEach(opt => {
      let cls = 'chip-low';
      if(opt.value === 'high'){
        cls = 'chip-high';
      }else if(opt.value === 'moderate'){
        cls = 'chip-medium';
      }
      frag.appendChild(createFilterChip(opt.label, 'risk', opt.value, cls));
    });
  }

  if(patientFilterOptions.wings.length){
    frag.appendChild(createFilterLabel('Wing'));
    patientFilterOptions.wings.forEach(opt => {
      const cls = opt.value === 'Unassigned' ? 'chip-neutral' : 'chip-wing';
      frag.appendChild(createFilterChip(opt.label, 'wing', opt.value, cls));
    });
  }

  container.appendChild(frag);
  updateFilterChipStates();
}

function createFilterLabel(text){
  const span = document.createElement('span');
  span.className = 'chip-group-label';
  span.textContent = text;
  return span;
}

function createFilterChip(label, type, value, extraClass){
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = `btn btn-sm chip ${extraClass || ''}`.trim();
  btn.dataset.filterType = type;
  btn.dataset.filterValue = value;
  btn.textContent = label;
  btn.addEventListener('click', () => applyFilter(type, value));
  return btn;
}

function applyFilter(type, value){
  if(type === 'all'){
    patientFilters = { facility: 'all', risk: 'all', wing: 'all' };
  }else{
    const isActive = patientFilters[type] === value;
    patientFilters[type] = isActive ? 'all' : value;
  }
  updateFilterChipStates();
  hideSuggestions();
  const selectionChanged = filterPatients();
  renderPatientList({ resetScroll: true, force: true });
  if(!filteredPatients.length){
    resetPatientCard();
    afterPatientChanged();
    return;
  }
  updatePatientDetails();
  ensureSelectedVisible();
  if(selectionChanged){
    afterPatientChanged();
  }
}

function updateFilterChipStates(){
  const container = document.getElementById('patientFilterChips');
  if(!container){
    return;
  }
  const buttons = container.querySelectorAll('.chip');
  buttons.forEach(btn => {
    const type = btn.dataset.filterType;
    const value = btn.dataset.filterValue;
    let active = false;
    if(type === 'all'){
      active = patientFilters.facility === 'all' && patientFilters.risk === 'all' && patientFilters.wing === 'all';
    }else if(Object.prototype.hasOwnProperty.call(patientFilters, type)){
      active = patientFilters[type] === value;
    }
    btn.classList.toggle('chip-active', active);
  });
}

function ensurePatientQuickSelect(){
  if(patientQuickSelectEl){
    return patientQuickSelectEl;
  }
  const el = document.getElementById('patientQuickSelect');
  if(!el){
    return null;
  }
  patientQuickSelectEl = el;
  if(!el.dataset.bound){
    el.dataset.bound = '1';
    el.addEventListener('change', evt => {
      if(suppressQuickSelectChange){
        return;
      }
      const nextId = evt.target.value;
      if(nextId){
        selectPatient(nextId);
      }
    });
  }
  return patientQuickSelectEl;
}

function syncPatientQuickSelect(value){
  const select = ensurePatientQuickSelect();
  if(!select){
    return;
  }
  suppressQuickSelectChange = true;
  const stringValue = value != null ? String(value) : '';
  const hasOption = Array.from(select.options).some(opt => opt.value === stringValue);
  select.value = hasOption ? stringValue : '';
  if(!hasOption && select.options.length){
    select.selectedIndex = 0;
  }
  suppressQuickSelectChange = false;
}

function refreshPatientQuickSelect(){
  const select = ensurePatientQuickSelect();
  if(!select){
    return;
  }
  suppressQuickSelectChange = true;
  select.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = filteredPatients.length ? 'Select a patient...' : 'No patients available';
  placeholder.disabled = !filteredPatients.length;
  placeholder.hidden = !!filteredPatients.length;
  select.appendChild(placeholder);
  filteredPatients.forEach(patient => {
    const option = document.createElement('option');
    option.value = String(patient.id);
    option.textContent = patient.name || `Patient ${patient.id}`;
    select.appendChild(option);
  });
  syncPatientQuickSelect(selectedPatientId);
  suppressQuickSelectChange = false;
}

function bindPatientSearch(){
  if(patientSearchInputEl && !patientSearchInputEl.dataset.bound){
    patientSearchInputEl.dataset.bound = '1';
    patientSearchInputEl.setAttribute('role', 'combobox');
    patientSearchInputEl.setAttribute('aria-expanded', 'false');
    patientSearchInputEl.addEventListener('input', () => {
      patientSearchTerm = patientSearchInputEl.value.trim().toLowerCase();
      const selectionChanged = filterPatients();
      renderPatientList({ resetScroll: true, force: true });
      if(filteredPatients.length){
        updatePatientDetails();
        if(selectionChanged){
          afterPatientChanged();
        }else{
          ensureSelectedVisible();
        }
      }else{
        resetPatientCard();
        afterPatientChanged();
      }
      updateSuggestions();
      patientSearchInputEl.setAttribute('aria-expanded', currentSuggestions.length ? 'true' : 'false');
    });
    patientSearchInputEl.addEventListener('keydown', handleSearchKeyDown);
    patientSearchInputEl.addEventListener('focus', () => {
      updateSuggestions();
      if(currentSuggestions.length){
        renderSuggestions();
        patientSuggestionsEl.classList.remove('d-none');
        patientSearchInputEl.setAttribute('aria-expanded', 'true');
      }
    });
    patientSearchInputEl.addEventListener('blur', () => {
      setTimeout(() => hideSuggestions(), 120);
    });
  }
  if(patientSuggestionsEl && !patientSuggestionsEl.dataset.bound){
    patientSuggestionsEl.dataset.bound = '1';
    patientSuggestionsEl.addEventListener('mousedown', handleSuggestionMouseDown);
    patientSuggestionsEl.addEventListener('mousemove', handleSuggestionMouseMove);
  }
}

function handleSearchKeyDown(evt){
  if(evt.key === 'ArrowDown'){
    if(!currentSuggestions.length){
      updateSuggestions();
    }
    moveSuggestion(1);
    evt.preventDefault();
  }else if(evt.key === 'ArrowUp'){
    if(!currentSuggestions.length){
      updateSuggestions();
    }
    moveSuggestion(-1);
    evt.preventDefault();
  }else if(evt.key === 'Enter'){
    if(currentSuggestions.length){
      evt.preventDefault();
      const idx = suggestionIndex >= 0 ? suggestionIndex : 0;
      selectSuggestionByIndex(idx);
    }else if(filteredPatients.length){
      evt.preventDefault();
      selectPatient(filteredPatients[0].id);
    }
  }else if(evt.key === 'Escape'){
    hideSuggestions();
  }
}

function moveSuggestion(delta){
  if(!currentSuggestions.length){
    return;
  }
  suggestionIndex = (suggestionIndex + delta + currentSuggestions.length) % currentSuggestions.length;
  renderSuggestions();
}

function selectSuggestionByIndex(index){
  if(!currentSuggestions.length){
    return;
  }
  const safeIndex = Math.max(0, Math.min(index, currentSuggestions.length - 1));
  const patient = currentSuggestions[safeIndex];
  if(!patient){
    return;
  }
  if(patientSearchInputEl){
    patientSearchInputEl.value = patient.name || '';
    patientSearchTerm = patientSearchInputEl.value.trim().toLowerCase();
  }
  hideSuggestions();
  selectPatient(patient.id);
}

function handleSuggestionMouseDown(evt){
  const item = evt.target.closest('.patient-suggest-item');
  if(!item){
    return;
  }
  evt.preventDefault();
  const id = item.dataset.patientId;
  const idx = currentSuggestions.findIndex(p => String(p.id) === String(id));
  selectSuggestionByIndex(idx >= 0 ? idx : 0);
}

function handleSuggestionMouseMove(evt){
  const item = evt.target.closest('.patient-suggest-item');
  if(!item){
    return;
  }
  const idx = currentSuggestions.findIndex(p => String(p.id) === String(item.dataset.patientId));
  if(idx >= 0 && idx !== suggestionIndex){
    suggestionIndex = idx;
    renderSuggestions();
  }
}

function updateSuggestions(){
  if(!patientSearchInputEl || !patientSearchTerm){
    hideSuggestions();
    return;
  }
  const matches = filteredPatients
    .map(p => ({ patient: p, score: scorePatientMatch(p, patientSearchTerm) }))
    .filter(entry => entry.score !== Number.POSITIVE_INFINITY)
    .sort((a, b) => a.score - b.score)
    .slice(0, MAX_PATIENT_SUGGESTIONS)
    .map(entry => entry.patient);

  currentSuggestions = matches;
  suggestionIndex = matches.length ? Math.min(Math.max(suggestionIndex, 0), matches.length - 1) : -1;

  if(!matches.length){
    hideSuggestions();
    return;
  }

  renderSuggestions();
}

function renderSuggestions(){
  if(!patientSuggestionsEl){
    return;
  }
  patientSuggestionsEl.innerHTML = '';
  if(!currentSuggestions.length){
    patientSuggestionsEl.classList.add('d-none');
    if(patientSearchInputEl){
      patientSearchInputEl.setAttribute('aria-expanded', 'false');
    }
    return;
  }
  const frag = document.createDocumentFragment();
  currentSuggestions.forEach((patient, idx) => {
    const li = document.createElement('li');
    li.className = 'patient-suggest-item' + (idx === suggestionIndex ? ' active' : '');
    li.dataset.patientId = String(patient.id);

    const primary = document.createElement('div');
    primary.className = 'patient-suggest-primary';
    primary.innerHTML = highlightMatch(patient.name || 'Unnamed patient', patientSearchTerm);

    const secondary = document.createElement('div');
    secondary.className = 'patient-suggest-secondary';
    const metaParts = [];
    if(patient.bed_id){
      metaParts.push(`Bed ${patient.bed_id}`);
    }
    if(patient.facility_id != null){
      metaParts.push(`Facility ${patient.facility_id}`);
    }
    if(patient.risk_level){
      metaParts.push(`Risk ${titleCase(patient.risk_level)}`);
    }
    secondary.textContent = metaParts.join(' | ');

    li.appendChild(primary);
    if(metaParts.length){
      li.appendChild(secondary);
    }
    frag.appendChild(li);
  });
  patientSuggestionsEl.appendChild(frag);
  patientSuggestionsEl.classList.remove('d-none');
  if(patientSearchInputEl){
    patientSearchInputEl.setAttribute('aria-expanded', 'true');
  }
}

function hideSuggestions(){
  if(patientSuggestionsEl){
    patientSuggestionsEl.classList.add('d-none');
    patientSuggestionsEl.innerHTML = '';
  }
  currentSuggestions = [];
  suggestionIndex = -1;
  if(patientSearchInputEl){
    patientSearchInputEl.setAttribute('aria-expanded', 'false');
  }
}

function filterPatients(){
  const previous = selectedPatientId;
  let dataset = patientsCache.filter(patientMatchesFilters).sort(patientComparator);
  if(patientSearchTerm){
    dataset = dataset.filter(p => patientHaystack(p).includes(patientSearchTerm));
  }
  filteredPatients = dataset;
  refreshPatientQuickSelect();
  setText('patientCountBadge', dataset.length ? `${dataset.length} in view` : '0 in view');
  setText('heroPatientCount', dataset.length ? String(dataset.length) : '0');

  if(!dataset.length){
    selectedPatientId = null;
  }else if(!selectedPatientId || !dataset.some(p => String(p.id) === String(selectedPatientId))){
    selectedPatientId = String(dataset[0].id);
  }

  buildVirtualRows();
  updateSuggestions();
  return previous !== selectedPatientId;
}

function buildVirtualRows(){
  patientRowMeta = [];
  patientRowsTotalHeight = 0;
  if(!filteredPatients.length){
    return;
  }
  let lastLabel = null;
  filteredPatients.forEach(patient => {
    const label = getGroupLabel(patient);
    if(label && label !== lastLabel){
      patientRowMeta.push({
        type: 'header',
        label,
        start: patientRowsTotalHeight,
        height: PATIENT_GROUP_HEIGHT,
        end: patientRowsTotalHeight + PATIENT_GROUP_HEIGHT
      });
      patientRowsTotalHeight += PATIENT_GROUP_HEIGHT;
      lastLabel = label;
    }
    patientRowMeta.push({
      type: 'patient',
      patient,
      start: patientRowsTotalHeight,
      height: PATIENT_ROW_HEIGHT,
      end: patientRowsTotalHeight + PATIENT_ROW_HEIGHT
    });
    patientRowsTotalHeight += PATIENT_ROW_HEIGHT;
  });
}

function getPatientInitials(name){
  if(!name) return 'PT';
  const parts = name.trim().split(/\s+/);
  if(parts.length === 1) return parts[0].charAt(0).toUpperCase();
  return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
}

function getRiskLevelClass(level){
  const normalized = (level || '').toString().toLowerCase();
  if(normalized === 'high') return 'risk-high';
  if(normalized === 'moderate') return 'risk-medium';
  if(normalized === 'low') return 'risk-low';
  return 'risk-unknown';
}

function renderPatientList(opts = {}){
  const { resetScroll = false, force = false, fromScroll = false } = opts;
  const list = document.getElementById('patientList');
  const empty = document.getElementById('patientListEmpty');
  if(!list || !empty){
    return;
  }

  if(!patientRowMeta.length){
    list.innerHTML = '';
    list.classList.remove('virtualized');
    list.style.height = '0px';
    empty.textContent = patientSearchTerm ? 'No patients match your search.' : 'No patients available.';
    empty.classList.remove('d-none');
    return;
  }

  empty.classList.add('d-none');
  list.classList.add('virtualized');
  list.style.height = `${patientRowsTotalHeight}px`;

  if(resetScroll && patientListWrapperEl){
    patientListWrapperEl.scrollTop = 0;
  }

  const scrollTop = patientListWrapperEl ? patientListWrapperEl.scrollTop : 0;
  const viewportHeight = patientListWrapperEl ? patientListWrapperEl.clientHeight : patientRowsTotalHeight;
  const viewportBottom = scrollTop + viewportHeight;
  const buffer = 160;

  let startIndex = 0;
  for(let i = 0; i < patientRowMeta.length; i += 1){
    if(patientRowMeta[i].end >= scrollTop - buffer){
      startIndex = i;
      break;
    }
  }

  let endIndex = patientRowMeta.length - 1;
  for(let i = startIndex; i < patientRowMeta.length; i += 1){
    if(patientRowMeta[i].start > viewportBottom + buffer){
      endIndex = Math.max(startIndex, i - 1);
      break;
    }
  }
  if(endIndex < startIndex){
    endIndex = startIndex;
  }

  list.innerHTML = '';
  const frag = document.createDocumentFragment();
  for(let i = startIndex; i <= endIndex && i < patientRowMeta.length; i += 1){
    const meta = patientRowMeta[i];
    const node = meta.type === 'header' ? renderGroupNode(meta) : renderPatientNode(meta);
    node.style.position = 'absolute';
    node.style.top = `${meta.start}px`;
    node.style.left = '0';
    node.style.right = '0';
    node.style.height = `${meta.height}px`;
    frag.appendChild(node);
  }
  list.appendChild(frag);

  if(!fromScroll){
    ensureSelectedVisible();
  }
}

function renderGroupNode(meta){
  const li = document.createElement('li');
  li.className = 'patient-list-group';
  li.textContent = meta.label;
  return li;
}

function renderPatientNode(meta){
  const patient = meta.patient;
  const li = document.createElement('li');
  li.className = 'patient-list-item' + (String(patient.id) === String(selectedPatientId) ? ' active' : '');
  li.dataset.patientId = String(patient.id);
  li.tabIndex = 0;

  const inner = document.createElement('div');
  inner.className = 'patient-list-item-inner';

  const avatar = document.createElement('div');
  avatar.className = 'patient-list-avatar';
  if(patient.avatar_url && isLocalAssetUrl(patient.avatar_url)){
    avatar.style.backgroundImage = `url(${patient.avatar_url})`;
  }else{
    avatar.textContent = getPatientInitials(patient.name);
  }

  const infoWrap = document.createElement('div');
  infoWrap.className = 'patient-list-info';

  const nameLine = document.createElement('div');
  nameLine.className = 'd-flex align-items-center gap-2';

  const nameSpan = document.createElement('span');
  nameSpan.className = 'patient-list-name';
  nameSpan.textContent = patient.name || 'Unnamed patient';

  const riskBadge = document.createElement('span');
  riskBadge.className = 'badge patient-risk-badge ' + getRiskLevelClass(patient.risk_level);
  riskBadge.textContent = patient.risk_level || 'Unknown';

  nameLine.appendChild(nameSpan);
  nameLine.appendChild(riskBadge);

  const metaLine = document.createElement('div');
  metaLine.className = 'patient-list-meta text-muted small';
  const metaParts = [];
  if(patient.bed_id){
    metaParts.push(`Bed ${patient.bed_id}`);
  }
  if(patient.facility_id != null){
    metaParts.push(`Facility ${patient.facility_id}`);
  }
  if(patient.primary_condition){
    metaParts.push(patient.primary_condition);
  }
  metaLine.textContent = metaParts.join(' | ');

  infoWrap.appendChild(nameLine);
  infoWrap.appendChild(metaLine);

  const extra = document.createElement('div');
  extra.className = 'patient-list-extra text-muted small fw-semibold';
  if(typeof patient.computed_score === 'number'){
    extra.textContent = `${Math.round(patient.computed_score)}%`;
  }else if(patient.age != null){
    extra.textContent = `Age ${patient.age}`;
  }else{
    extra.textContent = '--';
  }

  inner.appendChild(avatar);
  inner.appendChild(infoWrap);
  inner.appendChild(extra);

  li.appendChild(inner);
  li.addEventListener('click', () => selectPatient(patient.id));
  li.addEventListener('keydown', evt => {
    if(evt.key === 'Enter' || evt.key === ' '){
      evt.preventDefault();
      selectPatient(patient.id);
    }
  });

  return li;
}

function ensureSelectedVisible(){
  if(!patientListWrapperEl || !selectedPatientId || !patientRowMeta.length){
    return;
  }
  const meta = patientRowMeta.find(row => row.type === 'patient' && String(row.patient.id) === String(selectedPatientId));
  if(!meta){
    return;
  }
  const top = meta.start;
  const bottom = meta.end;
  const viewTop = patientListWrapperEl.scrollTop;
  const viewBottom = viewTop + patientListWrapperEl.clientHeight;
  if(top < viewTop){
    patientListWrapperEl.scrollTop = Math.max(top - PATIENT_GROUP_HEIGHT, 0);
  }else if(bottom > viewBottom){
    patientListWrapperEl.scrollTop = bottom - patientListWrapperEl.clientHeight + 16;
  }
}

function patientMatchesFilters(patient){
  if(patientFilters.facility !== 'all' && String(patient.facility_id) !== patientFilters.facility){
    return false;
  }
  if(patientFilters.risk !== 'all' && (patient.risk_level || '').toLowerCase() !== patientFilters.risk){
    return false;
  }
  if(patientFilters.wing !== 'all' && deriveBedWing(patient.bed_id) !== patientFilters.wing){
    return false;
  }
  return true;
}

function deriveBedWing(bedId){
  if(!bedId){
    return 'Unassigned';
  }
  const match = String(bedId).match(/^([A-Za-z]+)/);
  return match ? match[1].toUpperCase() : 'Unassigned';
}

function patientComparator(a, b){
  const facilityA = a.facility_id != null ? Number(a.facility_id) : Number.POSITIVE_INFINITY;
  const facilityB = b.facility_id != null ? Number(b.facility_id) : Number.POSITIVE_INFINITY;
  if(facilityA !== facilityB){
    return facilityA - facilityB;
  }
  const riskRank = { high: 0, moderate: 1, low: 2 };
  const riskA = riskRank[(a.risk_level || '').toLowerCase()] ?? 3;
  const riskB = riskRank[(b.risk_level || '').toLowerCase()] ?? 3;
  if(riskA !== riskB){
    return riskA - riskB;
  }
  const nameA = (a.name || '').toLowerCase();
  const nameB = (b.name || '').toLowerCase();
  if(nameA < nameB){
    return -1;
  }
  if(nameA > nameB){
    return 1;
  }
  return 0;
}

function scorePatientMatch(patient, term){
  const index = patientHaystack(patient).indexOf(term);
  return index === -1 ? Number.POSITIVE_INFINITY : index;
}

function patientHaystack(patient){
  return [
    patient.name || '',
    patient.bed_id || '',
    patient.primary_condition || '',
    patient.risk_level || '',
    patient.facility_id != null ? `facility ${patient.facility_id}` : ''
  ].join(' ').toLowerCase();
}

function getGroupLabel(patient){
  if(patientGroupMode === 'none'){
    return null;
  }
  if(patientGroupMode === 'risk'){
    return `Risk ${titleCase(patient.risk_level || 'Unknown')}`;
  }
  if(patient.facility_id != null){
    return `Facility ${patient.facility_id}`;
  }
  return 'Facility Unassigned';
}

function highlightMatch(text, term){
  const source = text || '';
  const lower = source.toLowerCase();
  const idx = lower.indexOf(term);
  if(idx === -1){
    return escapeHtml(source);
  }
  const end = idx + term.length;
  return `${escapeHtml(source.slice(0, idx))}<mark>${escapeHtml(source.slice(idx, end))}</mark>${escapeHtml(source.slice(end))}`;
}

function escapeHtml(value){
  const str = String(value ?? '');
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" };
  return str.replace(/[&<>"']/g, ch => map[ch] || ch);
}

function titleCase(value){
  if(!value){
    return '';
  }
  const str = value.toString().toLowerCase();
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function selectPatient(id){
  const idStr = id != null ? String(id) : null;
  if(!idStr || idStr === String(selectedPatientId)){
    hideSuggestions();
    ensureSelectedVisible();
    return;
  }
  selectedPatientId = idStr;
  syncPatientQuickSelect(idStr);
  updatePatientDetails();
  ensureSelectedVisible();
  hideSuggestions();
  if(commandPaletteOpen){
    closeCommandPalette();
  }
  afterPatientChanged();
}

function resetPatientCard(){
  currentPatient = null;
  syncPatientQuickSelect(null);
  setText('heroPatientContext', 'Select a patient to view details');
  setText('heroLastUpdated', '--');
  setText('healthTrendLabel', 'Live');
  setText('healthStatusLabel', 'Optimal');
  setText('heroVitalsContext', 'Sync devices to refresh vitals');
  setText('heroMoodValue', '--');
  setText('healthDonutCenter', '--');
  setText('heroMoodContext', 'Awaiting model prediction');
  resetPatientDrawer();
  setText('patientName', 'Select a patient');
  setText('patientOverallScore', '--');
  setText('patientOverallScoreCard', '--');
  setText('patientMovementStatus', '--');
  setText('patientEnvironmentStatus', '--');
  const ageBadge = document.getElementById('patientAgeBadge');
  if(ageBadge){
    ageBadge.textContent = 'Age: --';
  }
  const idBadge = document.getElementById('patientIdBadge');
  if(idBadge){
    idBadge.textContent = 'ID: --';
  }
  updateRiskBadge('--');
  setText('patientCondition', '--');
  setText('patientAllergy', '--');
  setText('patientCareFocus', '--');
  setText('patientBedLabel', 'Pick a patient to load assignments.');
  setText('patientLatestUpdate', '--');
  setText('patientLatestVitals', '--');
  setText('predictedMoodValue', '--');
  const avatar = document.getElementById('patientAvatar');
  if(avatar){
    const fallback = avatar.dataset.defaultSrc || avatar.getAttribute('data-default-src') || avatar.src;
    avatar.src = fallback;
    avatar.alt = 'Patient avatar';
  }
  updatePatientEmptyStates();
}

function updatePatientEmptyStates(){
  const trendsEmpty = document.getElementById('trendsEmpty');
  if(trendsEmpty){
    trendsEmpty.classList.toggle('d-none', Boolean(selectedPatientId));
  }
}

function applyPatientCard(patient){
  if(!patient){
    resetPatientCard();
    return;
  }
  currentPatient = patient;
  updatePatientDrawer(patient);
  const name = patient.name || 'Select a patient';
  setText('patientName', name);
  const ageBadge = document.getElementById('patientAgeBadge');
  if(ageBadge){
    const ageVal = Number.parseInt(patient.age, 10);
    ageBadge.textContent = Number.isFinite(ageVal) ? `Age: ${ageVal}` : 'Age: --';
  }
  const idBadge = document.getElementById('patientIdBadge');
  if(idBadge){
    const derivedId = patient.id_code || (patient.id ? `NS-${String(patient.id).padStart(4, '0')}` : '--');
    idBadge.textContent = `ID: ${derivedId}`;
  }
  updateRiskBadge(patient.risk_level || '--');
  const bedLabel = document.getElementById('patientBedLabel');
  if(bedLabel){
    const parts = [];
    if(patient.bed_id) parts.push(`Bed ${patient.bed_id}`);
    if(patient.facility_id) parts.push(`Facility ${patient.facility_id}`);
    bedLabel.textContent = parts.length ? parts.join(' | ') : 'No bed assignment';
  }
  setText('patientCondition', patient.primary_condition || 'No primary condition recorded');
  setText('patientAllergy', patient.allergies || 'No known allergies');
  setText('patientCareFocus', patient.care_focus || 'Personalized plan pending');
  const movementLabel = patient.movement_label || patient.movementStatus || '--';
  const environmentLabel = patient.environment_label || patient.currentAirQuality || '--';
  const overallScore = typeof patient.computed_score === 'number' ? `${Math.round(patient.computed_score)}%` : '--';
  setText('patientMovementStatus', movementLabel);
  setText('patientEnvironmentStatus', environmentLabel);
  setText('patientOverallScore', overallScore);
  setText('patientOverallScoreCard', overallScore);
  setText('healthTrendLabel', movementLabel && movementLabel !== '--' ? movementLabel : 'Loading');
  setText('healthStatusLabel', environmentLabel && environmentLabel !== '--' ? environmentLabel : 'Loading');
  setText('patientOverallScoreCard', overallScore);
  setText('healthDonutCenter', overallScore);
  const loadingMessage = name ? `Loading metrics for ${name}` : 'Loading patient metrics';
  setText('heroPatientContext', loadingMessage);
  setText('patientLatestUpdate', '--');
  setText('patientLatestVitals', '--');
  setText('predictedMoodValue', '--');
  const avatar = document.getElementById('patientAvatar');
  if(avatar){
    const fallback = avatar.dataset.defaultSrc || avatar.getAttribute('data-default-src') || avatar.src;
    const avatarUrl = (patient.avatar_url && isLocalAssetUrl(patient.avatar_url))
      ? patient.avatar_url
      : (patient.name ? buildAvatarUrl(patient.name) : null) || fallback;
    avatar.src = avatarUrl;
    avatar.alt = patient.name ? `${patient.name} avatar` : 'Patient avatar';
  }
}

function updatePatientDetails(){
  if(!selectedPatientId){
    resetPatientCard();
    renderPatientList({ force: true });
    return;
  }
  const found = patientsCache.find(p => String(p.id) === String(selectedPatientId));
  if(!found){
    resetPatientCard();
    renderPatientList({ force: true });
    return;
  }
  applyPatientCard(found);
  renderPatientList({ force: true });
}

function afterPatientChanged(){
  initFacilityUi();
  if(currentUser && (currentUser.id === 'super_admin' || currentUser.id === 'facility_admin')){
    initAdminUi();
  }
  loadPatientBundle();
  updatePatientEmptyStates();
}

// ----- Command Palette -----
function initCommandPalette(){
  commandPaletteEl = document.getElementById('commandPalette');
  if(!commandPaletteEl || commandPaletteEl.dataset.bound){
    return;
  }
  commandPaletteEl.dataset.bound = '1';
  commandPaletteInput = document.getElementById('commandPaletteInput');
  commandPaletteList = document.getElementById('commandPaletteList');
  const backdrop = commandPaletteEl.querySelector('.command-palette-backdrop');
  commandPaletteEl.setAttribute('aria-hidden', 'true');
  document.addEventListener('keydown', handleGlobalCommandHotkeys);
  if(backdrop){
    backdrop.addEventListener('click', () => closeCommandPalette());
  }
  if(commandPaletteEl){
    commandPaletteEl.addEventListener('click', evt => {
      if(evt.target === commandPaletteEl){
        closeCommandPalette();
      }
    });
  }
  if(commandPaletteInput){
    commandPaletteInput.addEventListener('input', () => updateCommandPaletteMatches(commandPaletteInput.value));
    commandPaletteInput.addEventListener('keydown', handleCommandPaletteKeyDown);
  }
  if(commandPaletteList){
    commandPaletteList.addEventListener('mousedown', handleCommandPaletteMouseDown);
    commandPaletteList.addEventListener('mousemove', handleCommandPaletteMouseMove);
  }
}

function handleGlobalCommandHotkeys(evt){
  if(commandPaletteOpen && evt.key === 'Escape'){
    evt.preventDefault();
    closeCommandPalette();
    return;
  }
  const target = evt.target;
  const tag = target && target.tagName ? target.tagName.toUpperCase() : '';
  const isEditable = target && (target.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT');
  if(evt.key === '/' && !evt.metaKey && !evt.ctrlKey && !evt.altKey){
    if(isEditable){
      return;
    }
    evt.preventDefault();
    openCommandPalette('');
    return;
  }
  if((evt.key === 'k' || evt.key === 'K') && (evt.metaKey || evt.ctrlKey)){
    evt.preventDefault();
    openCommandPalette('');
  }
}

function openCommandPalette(prefill){
  if(!commandPaletteEl){
    return;
  }
  if(commandPaletteOpen){
    if(commandPaletteInput){
      commandPaletteInput.focus({ preventScroll: true });
      commandPaletteInput.select();
    }
    return;
  }
  commandPaletteOpen = true;
  commandPaletteReturnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  commandPaletteEl.classList.add('open');
  commandPaletteEl.setAttribute('aria-hidden', 'false');
  document.body.classList.add('command-palette-open');
  if(commandPaletteFocusCleanup){
    commandPaletteFocusCleanup();
  }
  commandPaletteFocusCleanup = trapFocus(commandPaletteEl);
  const value = prefill || '';
  if(commandPaletteInput){
    commandPaletteInput.value = value;
    commandPaletteInput.setAttribute('aria-expanded', 'false');
    setTimeout(() => commandPaletteInput.focus({ preventScroll: true }), 0);
  }
  hideSuggestions();
  updateCommandPaletteMatches(value);
}

function closeCommandPalette(){
  if(!commandPaletteOpen || !commandPaletteEl){
    return;
  }
  commandPaletteOpen = false;
  commandPaletteEl.classList.remove('open');
  commandPaletteEl.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('command-palette-open');
  if(commandPaletteFocusCleanup){
    commandPaletteFocusCleanup();
    commandPaletteFocusCleanup = null;
  }
  if(commandPaletteInput){
    commandPaletteInput.value = '';
    commandPaletteInput.setAttribute('aria-expanded', 'false');
  }
  commandPaletteMatches = [];
  commandPaletteIndex = -1;
  if(commandPaletteList){
    commandPaletteList.innerHTML = '';
  }
  const focusTarget = commandPaletteReturnFocus;
  commandPaletteReturnFocus = null;
  if(focusTarget && typeof focusTarget.focus === 'function'){
    focusTarget.focus({ preventScroll: true });
  }
}

function updateCommandPaletteMatches(rawTerm){
  const term = (rawTerm || '').trim().toLowerCase();
  commandPaletteMatches = buildCommandPaletteMatches(term);
  commandPaletteIndex = commandPaletteMatches.length ? 0 : -1;
  renderCommandPaletteMatches(term);
  if(commandPaletteInput){
    commandPaletteInput.setAttribute('aria-expanded', commandPaletteMatches.length ? 'true' : 'false');
  }
}

function buildCommandPaletteMatches(term){
  const matches = [];
  const lowered = term || '';
  const includeAll = lowered.length === 0;
  commandPaletteActions.forEach((action, idx) => {
    const label = action.label.toLowerCase();
    const subtitle = (action.subtitle || '').toLowerCase();
    let score = Number.POSITIVE_INFINITY;
    if(includeAll){
      score = idx;
    }else{
      const inLabel = label.indexOf(lowered);
      const inSubtitle = subtitle.indexOf(lowered);
      score = Math.min(inLabel >= 0 ? inLabel : Number.POSITIVE_INFINITY, inSubtitle >= 0 ? inSubtitle : Number.POSITIVE_INFINITY);
    }
    if(score !== Number.POSITIVE_INFINITY){
      matches.push({ type: 'action', score, action });
    }else if(includeAll){
      matches.push({ type: 'action', score: idx, action });
    }
  });
  const seen = new Set();
  if(includeAll && currentPatient && currentPatient.id != null){
    seen.add(String(currentPatient.id));
    matches.push({ type: 'patient', score: commandPaletteActions.length, patient: currentPatient });
  }
  const patients = Array.isArray(patientsCache) ? patientsCache : [];
  if(includeAll){
    const limit = Math.max(0, MAX_COMMAND_RESULTS - matches.length);
    if(limit > 0){
      const sorted = [...patients].sort(patientComparator);
      for(const patient of sorted){
        const pid = patient.id != null ? String(patient.id) : null;
        if(pid && seen.has(pid)){
          continue;
        }
        matches.push({ type: 'patient', score: matches.length + 10, patient });
        if(pid){
          seen.add(pid);
        }
        if(matches.length >= MAX_COMMAND_RESULTS){
          break;
        }
      }
    }
  }else{
    patients.forEach(patient => {
      const score = scorePatientMatch(patient, lowered);
      if(score === Number.POSITIVE_INFINITY){
        return;
      }
      const pid = patient.id != null ? String(patient.id) : `anon-${score}-${Math.random()}`;
      if(seen.has(pid)){
        return;
      }
      seen.add(pid);
      matches.push({ type: 'patient', score, patient });
    });
  }
  matches.sort((a, b) => a.score - b.score);
  return matches.slice(0, MAX_COMMAND_RESULTS);
}

function renderCommandPaletteMatches(term){
  if(!commandPaletteList){
    return;
  }
  commandPaletteList.innerHTML = '';
  if(!commandPaletteMatches.length){
    const empty = document.createElement('li');
    empty.className = 'command-palette-empty';
    empty.textContent = term ? 'No results found.' : 'Start typing to search patients or actions.';
    commandPaletteList.appendChild(empty);
    return;
  }
  commandPaletteMatches.forEach((match, idx) => {
    const li = document.createElement('li');
    li.dataset.index = String(idx);
    li.dataset.type = match.type;
    li.className = 'command-palette-item';
    const primary = document.createElement('div');
    primary.className = 'command-palette-primary';
    const secondary = document.createElement('div');
    secondary.className = 'command-palette-secondary';
    if(match.type === 'action'){
      primary.innerHTML = highlightCommandText(match.action.label, term);
      secondary.textContent = match.action.subtitle || '';
    }else{
      const patient = match.patient || {};
      primary.innerHTML = highlightMatch(patient.name || 'Unnamed patient', term);
      const meta = [];
      if(patient.bed_id){
        meta.push(`Bed ${patient.bed_id}`);
      }
      if(patient.facility_id != null){
        meta.push(`Facility ${patient.facility_id}`);
      }
      if(patient.risk_level){
        meta.push(`Risk ${patient.risk_level}`);
      }
      secondary.textContent = meta.join(' | ');
    }
    li.appendChild(primary);
    if(secondary.textContent){
      li.appendChild(secondary);
    }
    commandPaletteList.appendChild(li);
  });
  updateCommandPaletteActive();
}

function highlightCommandText(text, term){
  if(!term){
    return escapeHtml(text || '');
  }
  return highlightMatch(text || '', term);
}

function updateCommandPaletteActive(){
  if(!commandPaletteList){
    return;
  }
  const children = Array.from(commandPaletteList.children);
  children.forEach((child, idx) => {
    child.classList.toggle('active', idx === commandPaletteIndex);
  });
  if(commandPaletteIndex >= 0 && commandPaletteIndex < children.length){
    children[commandPaletteIndex].scrollIntoView({ block: 'nearest' });
  }
}

function moveCommandPaletteSelection(delta){
  if(!commandPaletteMatches.length){
    return;
  }
  commandPaletteIndex = (commandPaletteIndex + delta + commandPaletteMatches.length) % commandPaletteMatches.length;
  updateCommandPaletteActive();
}

function executeCommandPaletteMatch(match){
  if(!match){
    return;
  }
  closeCommandPalette();
  if(match.type === 'action' && match.action){
    setTimeout(() => match.action.run(), 0);
  }else if(match.type === 'patient' && match.patient){
    setTimeout(() => selectPatient(match.patient.id), 0);
  }
}

function handleCommandPaletteKeyDown(evt){
  if(evt.key === 'ArrowDown'){
    moveCommandPaletteSelection(1);
    evt.preventDefault();
  }else if(evt.key === 'ArrowUp'){
    moveCommandPaletteSelection(-1);
    evt.preventDefault();
  }else if(evt.key === 'Enter'){
    if(commandPaletteIndex >= 0 && commandPaletteIndex < commandPaletteMatches.length){
      evt.preventDefault();
      executeCommandPaletteMatch(commandPaletteMatches[commandPaletteIndex]);
    }
  }else if(evt.key === 'Escape'){
    evt.preventDefault();
    closeCommandPalette();
  }
}

function handleCommandPaletteMouseDown(evt){
  const item = evt.target.closest('li');
  if(!item || item.classList.contains('command-palette-empty')){
    return;
  }
  evt.preventDefault();
  const idx = Number(item.dataset.index);
  if(Number.isInteger(idx) && idx >= 0 && idx < commandPaletteMatches.length){
    commandPaletteIndex = idx;
    updateCommandPaletteActive();
    executeCommandPaletteMatch(commandPaletteMatches[idx]);
  }
}

function handleCommandPaletteMouseMove(evt){
  const item = evt.target.closest('li');
  if(!item || item.classList.contains('command-palette-empty')){
    return;
  }
  const idx = Number(item.dataset.index);
  if(!Number.isInteger(idx) || idx === commandPaletteIndex){
    return;
  }
  commandPaletteIndex = idx;
  updateCommandPaletteActive();
}

function populateProfileModal(me) {
  const rawRole = me && me.role ? me.role : '';
  refreshUserAvatar(me);
  const displayName = (me && (me.name || me.email)) || '--';
  setText('profileName', displayName);
  setText('profileEmail', me && me.email ? me.email : '--');
  const niceRole = rawRole ? rawRole.replace(/_/g, ' ').split(' ').map((word) => titleCase(word)).join(' ') : '--';
  setText('profileRole', niceRole);
  const roleInfo = roles[rawRole] || {};
  const summaryEl = document.getElementById('profileRoleSummary');
  if(summaryEl){
    const description = roleInfo.description || '--';
    summaryEl.textContent = description;
  }
  const facilityLabel = (me && me.facility_id !== null && me.facility_id !== undefined)
    ? `Facility ${me.facility_id}`
    : (rawRole === 'super_admin' ? 'All Facilities' : 'Not assigned');
  setText('profileFacility', facilityLabel);
  const assignedWrap = document.getElementById('profileAssigned');
  if (assignedWrap) {
    assignedWrap.innerHTML = '';
    const assigned = Array.isArray(me && me.assigned_patient_ids) ? me.assigned_patient_ids : [];
    if (!assigned.length) {
      assignedWrap.classList.add('empty');
      assignedWrap.textContent = 'No specific assignments';
    } else {
      assignedWrap.classList.remove('empty');
      assigned.forEach((id) => {
        const span = document.createElement('span');
        span.className = 'badge';
        span.textContent = `Patient ${id}`;
        assignedWrap.appendChild(span);
      });
    }
  }
}

function initProfileModal() {

  const link = document.getElementById('profileViewLink');
  const modalEl = document.getElementById('profileModal');
  if(!link || !modalEl || link.dataset.bound){
    return;
  }
  link.dataset.bound = '1';
  link.addEventListener('click', async (evt) => {
    evt.preventDefault();
    try{
      if(!currentUserProfile){
        currentUserProfile = await apiGet('/api/me');
      }
    }catch(err){
      console.error('[dashboard] profile fetch error', err);
    }
    populateProfileModal(currentUserProfile || {});
    if(typeof bootstrap !== 'undefined'){
      const modalInstance = bootstrap.Modal.getOrCreateInstance(modalEl);
      modalInstance.show();
    }
  });
}
function initButtons(){
  const syncBtn = document.getElementById('syncWearable');
  if(syncBtn && !syncBtn.dataset.bound){
    syncBtn.dataset.bound = '1';
    syncBtn.addEventListener('click', async function(){
      const btn = this;
      btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Syncing...';
      btn.disabled = true;
      try{
        await loadPatientBundle();
        showToast({ title:'Sync complete', message:'Latest device readings have been refreshed.' });
      }catch(e){
        console.error('[dashboard] sync refresh failed', e);
        showToast({ title:'Sync failed', message:'Unable to refresh data right now.', variant:'error' });
      }finally{
        setTimeout(() => {
          btn.innerHTML = '<i class="fas fa-sync-alt me-2"></i> Sync Devices Now';
          btn.disabled = false;
        }, 1200);
      }
    });
  }
  const refreshCheckinsBtn = document.getElementById('refreshCheckinsBtn');
  if(refreshCheckinsBtn && !refreshCheckinsBtn.dataset.bound){
    refreshCheckinsBtn.dataset.bound = '1';
    refreshCheckinsBtn.addEventListener('click', () => loadCheckins());
  }
  const openPaletteBtn = document.getElementById('openCommandPaletteBtn');
  if(openPaletteBtn && !openPaletteBtn.dataset.bound){
    openPaletteBtn.dataset.bound = '1';
    openPaletteBtn.addEventListener('click', () => openCommandPalette(''));
  }
  const roleItems = document.querySelectorAll('.role-switcher-item');
  roleItems.forEach(item => {
    if(item.dataset.bound){
      return;
    }
    item.dataset.bound = '1';
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const roleId = item.getAttribute('data-role');
      if(roleId && roles[roleId]){
        switchRole(roleId);
      }
    });
  });
  const roleCards = document.querySelectorAll('.role-option[data-role]');
  roleCards.forEach(card => {
    if(card.dataset.bound){
      return;
    }
    card.dataset.bound = '1';
    card.addEventListener('click', () => {
      const roleId = card.getAttribute('data-role');
      if(roleId && roles[roleId]){
        switchRole(roleId);
        roleCards.forEach(el => el.classList.remove('active'));
        card.classList.add('active');
      }
    });
  });
}

function initGoalModal(){
  const modal = document.getElementById('goalModal');
  if(!modal || modal.dataset.bound){
    return;
  }
  modal.dataset.bound = '1';
  modal.addEventListener('shown.bs.modal', () => {
    const titleInput = document.getElementById('goalTitle');
    if(titleInput){
      titleInput.focus();
      titleInput.select();
    }
  });
}

function initForms(){
  const checkinFormEl = document.getElementById('checkinForm');
  if(checkinFormEl && !checkinFormEl.dataset.bound){
    checkinFormEl.dataset.bound = '1';
    checkinFormEl.addEventListener('submit', submitCheckin);
  }
  const goalFormEl = document.getElementById('goalForm');
  if(goalFormEl && !goalFormEl.dataset.bound){
    goalFormEl.dataset.bound = '1';
    goalFormEl.addEventListener('submit', submitGoal);
  }
}

function initSidebarNav(){
  const links = document.querySelectorAll('.sidebar-link');
  links.forEach(link => {
    link.addEventListener('click', e => {
      const target = link.getAttribute('href');
      if(target && target.startsWith('#')){
        links.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        const section = document.querySelector(target);
        if(section){
          section.scrollIntoView({ behavior:'smooth', block:'start' });
        }
        e.preventDefault();
      }
    });
  });
}












document.addEventListener('DOMContentLoaded', () => {
  console.log('[dashboard] DOMContentLoaded');
  initTheme();
  initAvatarUpload();
  initCharts();
  initButtons();
  initGoalModal();
  initForms();
  initSidebarNav();
  initSidebarToggle();
  initCommandPalette();
  initProfileModal();
  initPatientDrawer();
  initUser().then(() => {
    initPatientSelector();
    initFacilityUi();
    initAdminUi();
    updatePatientEmptyStates();
  }).catch(err => {
    console.error('[dashboard] initUser promise rejected', err);
    initPatientSelector();
    initFacilityUi();
    updatePatientEmptyStates();
  });
  setInterval(() => {
    if(document.hidden){
      return;
    }
    if(!selectedPatientId){
      return;
    }
    loadPatientBundle();
  }, 60000);
});























