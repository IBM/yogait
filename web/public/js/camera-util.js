
const isSafari = function () {
  const hasSafari = /Safari/i.test(navigator.userAgent)
  const hasChrome = /Chrome/i.test(navigator.userAgent)

  return hasSafari && !hasChrome
}

/**
 * Load the video camera
 *
 * @param {Node | String} domNode - DOM Node or id of the DOM Node to load video into (default: 'video')
 */
export async function loadVideo (domNode, size) {
  const video = await setupCamera(domNode, size)
  video.play()
  return video
}

/**
 * Set up the navigator media device
 *
 * @param {Node | String} domNode - DOM Node or id of the DOM Node to load video into (default: 'video')
 */
async function setupCamera (domNode = 'video', size) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Browser API navigator.mediaDevices.getUserMedia not available')
  }

  const video = typeof domNode === 'string' ? document.getElementById(domNode) : domNode

  video.width = size.width
  video.height = size.height
  let constraint = {
    'audio': false,
    'video': {
      facingMode: 'user'
    }
  }

  if (!isSafari()) {
    constraint.video.width = size.width
    constraint.video.height = size.height
  }

  video.srcObject = await navigator.mediaDevices.getUserMedia(constraint)

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video)
    }
  })
}

export const preferredVideoSize = function (video) {
  let size = {
    width: window.innerWidth,
    height: window.innerHeight
  }

  const w = Math.min(window.innerWidth, 1400)
  const h = Math.min(window.innerHeight, 1400)
  let vw = 800
  let vh = 600

  if (video) {
    vw = video.videoWidth
    vh = video.videoHeight
  }

  const videoRatio = vw / vh

  if (w / vw < h / vh) {
    let width = w < 400 ? w : (w < 600 ? w * 0.85 : w * 0.7)
    size = {
      width: width,
      height: width / videoRatio
    }
  } else {
    let height = h < 300 ? h : (h < 450 ? h * 0.85 : h * 0.7)
    size = {
      height: height,
      width: height * videoRatio
    }
  }

  return size
}
